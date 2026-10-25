"""Single-kernel peer pull and LSE merge for unfused CUSTOM all-to-all.

At most one CTA per SM, 16 warps each; a warp owns one output row (token,
local head) at a time and a CTA's warps take consecutive rows, so its reads of
every peer are contiguous. For a row the warp's elected lane issues one 1 KiB
bulk copy per rank straight from that rank's symmetric O into the warp's
shared-memory slot, while lanes 0..World-1 read the matching LSE. The merge
runs from shared memory into FP32 registers and writes the final BF16 row.
With 148 SMs every row of a decode batch up to 2368 rows is in flight at once;
larger batches loop over the single slot.

Synchronization: CTA0 publishes this rank's readiness to every peer, then every
CTA waits for all peers' readiness on local slots. No consumption barrier:
the layer's following TP collective orders every peer's next source write
after this rank's reads.
"""

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils

from .mla_helpers import wait_ready
from .ops import COUNT, EPOCH, READY

DIM = 512
ROW_BYTES = DIM * 2
NEG_INF = float("-inf")
NAN = float("nan")


class PeerPullMerge:
    def __init__(self, world: int, local_heads: int, heads: int, warps: int):
        self.world = world
        self.local_heads = local_heads
        self.heads = heads
        self.warps = warps
        self.threads = warps * 32
        # One 1 KiB-per-rank slot and one mbarrier per warp, the epoch.
        self.smem_bytes = warps * world * ROW_BYTES + warps * 8 + 8 + 128

    @cute.jit
    def __call__(self, peers: cute.Tensor, control_peers: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rank: Int32, grid_ctas: Int32, stream):
        self.kernel(peers, control_peers, output, rows, rank).launch(
            grid=(grid_ctas, 1, 1), block=(self.threads, 1, 1), smem=self.smem_bytes, stream=stream)

    @cute.jit
    def issue(self, peers: cute.Tensor, buffer, barriers, warp: Int32, lane: Int32,
              rank: Int32, row: Int32) -> Float32:
        """Copy every rank's 1 KiB O of this row into the warp's slot; return lane's LSE."""
        world = self.world
        token = row // self.local_heads
        head = row - token * self.local_heads
        source_row = Int64(token) * self.heads + Int64(rank) * self.local_heads + head
        mbar = barriers + warp
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, world * ROW_BYTES)
            for source in cutlass.range_constexpr(world):
                src = cute.make_tensor(
                    cute.make_ptr(cutlass.BFloat16, peers[source, 0], cute.AddressSpace.gmem,
                                  assumed_align=16) + source_row * DIM,
                    cute.make_layout((DIM,)))
                dst = cute.make_tensor(buffer + (warp * world + source) * DIM, cute.make_layout((DIM,)))
                cute.copy(cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.BFloat16,
                                              num_bits_per_copy=ROW_BYTES * 8),
                          src, dst, mbar_ptr=mbar)
        lse = Float32(NEG_INF)
        if lane < world:
            lse_ptr = cute.make_ptr(Float32, peers[lane, 1], cute.AddressSpace.gmem, assumed_align=4)
            lse = cute.arch.load((lse_ptr + source_row).llvm_ptr, Float32)
        return lse

    @cute.kernel
    def kernel(self, peers: cute.Tensor, control_peers: cute.Tensor, output: cute.Tensor,
               rows: Int32, rank: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_ctas, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        world = self.world

        smem = utils.SmemAllocator()
        # Per warp: a slot of World rows of 1 KiB, and its mbarrier.
        buffer = cute.recast_ptr(smem.allocate(self.warps * world * ROW_BYTES, 128), dtype=cutlass.BFloat16)
        barriers = cute.recast_ptr(smem.allocate(self.warps * 8, 8), dtype=Int64)
        epoch_slot = cute.make_tensor(cute.recast_ptr(smem.allocate(8, 8), dtype=Int64), cute.make_layout((1,)))

        control = cute.make_tensor(
            cute.make_ptr(Int64, control_peers[rank], cute.AddressSpace.gmem, assumed_align=8),
            cute.make_layout((READY + 32,)))

        if lane == 0:
            cute.arch.mbarrier_init(barriers + warp, 1)
        if tidx == 0:
            epoch_slot[0] = control[EPOCH] + 1
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        epoch = epoch_slot[0]

        # Readiness: one fixed CTA publishes, lanes 0..World-1 each to a peer.
        if bidx == 0 and tidx < world:
            ready = cute.make_ptr(Int64, control_peers[tidx], cute.AddressSpace.gmem, assumed_align=8)
            cute.arch.store((ready + READY + rank).llvm_ptr, epoch, sem="release", scope="sys")
        if tidx < world:
            wait_ready(control.iterator + READY + tidx, epoch)
        # Extends the polling lanes' acquire to every warp's later loads, and
        # orders it before the async-proxy bulk reads of peer records.
        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.global")

        local_heads = self.local_heads
        total_rows = rows * local_heads
        stride = grid_ctas * self.warps
        first = bidx * self.warps + warp

        phase = Int32(0)
        row = first
        lse = Float32(NEG_INF)
        if row < total_rows:
            lse = self.issue(peers, buffer, barriers, warp, lane, rank, row)
        while row < total_rows:
            # Weights from the lanes' LSE values; empty ranks carry -inf.
            maximum = cute.arch.warp_reduction_max(lse)
            has_keys = maximum != NEG_INF
            weight = Float32(0.0)
            if lse != NEG_INF:
                weight = cute.math.exp2(lse - maximum, fastmath=True)
            denominator = cute.arch.warp_reduction_sum(weight)
            nan = cute.arch.vote_any_sync(lse != lse)

            cute.arch.mbarrier_wait(barriers + warp, phase)
            # Each lane merges 16 consecutive features of the row.
            accumulator = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
            for i in cutlass.range_constexpr(16):
                accumulator[i] = Float32(0.0)
            for source in cutlass.range_constexpr(world):
                w = cute.arch.shuffle_sync(weight, source)
                if w != Float32(0.0):
                    values = cute.make_tensor(buffer + (warp * world + source) * DIM + lane * 16,
                                              cute.make_layout((16,)))
                    for i in cutlass.range_constexpr(16):
                        accumulator[i] = accumulator[i] + Float32(values[i]) * w
            scale = Float32(0.0)
            if has_keys:
                scale = cute.arch.rcp_approx(denominator)
            token = row // local_heads
            head = row - token * local_heads
            # 16 BF16 per lane: a 32-byte aligned chunk of the output row.
            offset = cute.assume((Int64(head) * rows + token) * DIM + lane * 16, divby=16)
            out = cute.make_tensor(
                cute.make_ptr(cutlass.BFloat16, output.iterator.toint() + offset * 2,
                              cute.AddressSpace.gmem, assumed_align=32),
                cute.make_layout((16,)))
            result = cute.make_rmem_tensor(cute.make_layout((16,)), cutlass.BFloat16)
            for i in cutlass.range_constexpr(16):
                value = accumulator[i] * scale
                if nan:
                    value = NAN
                result[i] = cutlass.BFloat16(value)
            cute.autovec_copy(result, out)
            # All lanes finished their generic reads of this slot before the
            # async proxy refills it.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            row = row + stride
            if row < total_rows:
                lse = self.issue(peers, buffer, barriers, warp, lane, rank, row)
            phase = phase ^ 1

        # Advance the epoch only after every CTA has read it.
        cute.arch.sync_threads()
        if tidx == 0:
            finished = cute.arch.atomic_add((control.iterator + COUNT).llvm_ptr, Int64(1),
                                            sem="acq_rel", scope="gpu")
            if finished == Int64(grid_ctas - 1):
                control[COUNT] = Int64(0)
                control[EPOCH] = epoch


def warps_for(world: int) -> int:
    """16 warps, or the largest multiple of 4 whose World 1 KiB slots fit in SMEM."""
    return min(16, (220 * 1024) // (world * ROW_BYTES) // 4 * 4)


@functools.cache
def compile_pull_merge(world: int, local_heads: int, heads: int, warps: int):
    kernel = PeerPullMerge(world, local_heads, heads, warps)
    peers = cute.runtime.make_fake_compact_tensor(Int64, (world, 2), stride_order=(1, 0), assumed_align=16)
    controls = cute.runtime.make_fake_compact_tensor(Int64, (world,), assumed_align=16)
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (local_heads, cute.sym_int(), DIM), stride_order=(2, 1, 0), assumed_align=16)
    return cute.compile(kernel, peers, controls, output, Int32(1), Int32(0), Int32(1),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 3")
