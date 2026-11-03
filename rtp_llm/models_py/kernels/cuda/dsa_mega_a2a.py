"""MegaA2A: LSE merge of DCP partial attention outputs, pulled from peers inside one kernel.

Each rank of the decode DCP (TP) group writes its partial output O [rows, record_heads, dim]
(BF16) and base-2 LSE [rows, record_heads] (FP32) into symmetric memory. One kernel then pulls,
for this rank's heads, every rank's O row (one bulk copy per rank) and LSE, and writes the
LSE-weighted merge [local_heads, rows, dim]. It replaces the pack / NCCL all-to-all / combine
of MlaDcpCommunicator.combine (DSA_DCP_COMBINE=mega_a2a).

Kernel (from the K3 FIA2A CUSTOM all-to-all): at most one CTA per SM, a warp per output row
(token, local head); CTA0 publishes this rank's readiness to every peer, every CTA waits for all
peers' readiness. No consumption barrier: the layer's following TP collective (the o_proj
reduce-scatter) orders every peer's next source write after this rank's reads, so serial
callers without such a collective must add one between calls.
"""

import functools

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import tvm_ffi
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

# Peer control layout, one int64 array per rank in symmetric memory. Epochs are monotonic per
# buffer (no reset); a receiver's ready[src] is written only by rank src.
EPOCH, COUNT, READY = 0, 1, 16
CONTROL_SIZE = READY + 32


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    # CuTe 4.4.2 has no arch.globaltimer wrapper. This register counts ns.
    return Int64(llvm.inline_asm(T.i64(), [], 'mov.u64 $0, %globaltimer;', '=l',
                                has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def timeout_trap(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'trap;', '', has_side_effects=True,
                    asm_dialect=0, loc=loc, ip=ip)


@cute.jit
def wait_ready(slot: cute.Pointer, epoch: Int64):
    """Acquire a local ready slot once it reaches epoch; trap after 5 s."""
    # Read the clock only after a failed poll: on sm_103a a timer read
    # scheduled between the address and its first load faulted.
    if cute.arch.load(slot.llvm_ptr, Int64, sem="acquire", scope="sys") < epoch:
        start = read_globaltimer()
        while cute.arch.load(slot.llvm_ptr, Int64, sem="acquire", scope="sys") < epoch:
            if read_globaltimer() - start > Int64(5000000000):
                timeout_trap()


NEG_INF = float("-inf")
NAN = float("nan")


class MegaA2AKernel:
    def __init__(self, world: int, local_heads: int, heads: int, warps: int, dim: int = 512):
        # Lanes split a row evenly, and a row stays a multiple of 16 bytes for the bulk copy.
        assert dim % 32 == 0
        self.world = world
        self.local_heads = local_heads
        self.heads = heads
        self.warps = warps
        self.dim = dim
        self.threads = warps * 32
        # One row-per-rank slot and one mbarrier per warp, the epoch.
        self.smem_bytes = warps * world * dim * 2 + warps * 8 + 8 + 128

    @cute.jit
    def __call__(self, peers: cute.Tensor, control_peers: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rank: Int32, grid_ctas: Int32, stream):
        self.kernel(peers, control_peers, output, rows, rank).launch(
            grid=(grid_ctas, 1, 1), block=(self.threads, 1, 1), smem=self.smem_bytes, stream=stream)

    @cute.jit
    def issue(self, peers: cute.Tensor, buffer, barriers, warp: Int32, lane: Int32,
              rank: Int32, row: Int32) -> Float32:
        """Copy every rank's O of this row into the warp's slot; return lane's LSE."""
        world, DIM = self.world, self.dim
        token = row // self.local_heads
        head = row - token * self.local_heads
        source_row = Int64(token) * self.heads + Int64(rank) * self.local_heads + head
        mbar = barriers + warp
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, world * DIM * 2)
            for source in cutlass.range_constexpr(world):
                src = cute.make_tensor(
                    cute.make_ptr(cutlass.BFloat16, peers[source, 0], cute.AddressSpace.gmem,
                                  assumed_align=16) + source_row * DIM,
                    cute.make_layout((DIM,)))
                dst = cute.make_tensor(buffer + (warp * world + source) * DIM, cute.make_layout((DIM,)))
                cute.copy(cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.BFloat16,
                                              num_bits_per_copy=DIM * 2 * 8),
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
        world, DIM = self.world, self.dim
        E = DIM // 32  # features merged per lane

        smem = utils.SmemAllocator()
        # Per warp: a slot of World rows, and its mbarrier.
        buffer = cute.recast_ptr(smem.allocate(self.warps * world * DIM * 2, 128), dtype=cutlass.BFloat16)
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
            # Each lane merges E consecutive features of the row.
            accumulator = cute.make_rmem_tensor(cute.make_layout((E,)), Float32)
            for i in cutlass.range_constexpr(E):
                accumulator[i] = Float32(0.0)
            for source in cutlass.range_constexpr(world):
                w = cute.arch.shuffle_sync(weight, source)
                if w != Float32(0.0):
                    values = cute.make_tensor(buffer + (warp * world + source) * DIM + lane * E,
                                              cute.make_layout((E,)))
                    for i in cutlass.range_constexpr(E):
                        accumulator[i] = accumulator[i] + Float32(values[i]) * w
            scale = Float32(0.0)
            if has_keys:
                scale = cute.arch.rcp_approx(denominator)
            token = row // local_heads
            head = row - token * local_heads
            # E BF16 per lane: a 2E-byte aligned chunk of the output row.
            offset = cute.assume((Int64(head) * rows + token) * DIM + lane * E, divby=E)
            out = cute.make_tensor(
                cute.make_ptr(cutlass.BFloat16, output.iterator.toint() + offset * 2,
                              cute.AddressSpace.gmem, assumed_align=2 * E),
                cute.make_layout((E,)))
            result = cute.make_rmem_tensor(cute.make_layout((E,)), cutlass.BFloat16)
            for i in cutlass.range_constexpr(E):
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


def warps_for(world: int, dim: int = 512) -> int:
    """16 warps, or the largest multiple of 4 whose World row slots fit in SMEM."""
    return min(16, (220 * 1024) // (world * dim * 2) // 4 * 4)


@functools.cache
def compile_mega_a2a(world: int, local_heads: int, heads: int, warps: int, dim: int = 512):
    kernel = MegaA2AKernel(world, local_heads, heads, warps, dim)
    peers = cute.runtime.make_fake_compact_tensor(Int64, (world, 2), stride_order=(1, 0), assumed_align=16)
    controls = cute.runtime.make_fake_compact_tensor(Int64, (world,), assumed_align=16)
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (local_heads, cute.sym_int(), dim), stride_order=(2, 1, 0), assumed_align=16)
    return cute.compile(kernel, peers, controls, output, Int32(1), Int32(0), Int32(1),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 3")


class MegaA2ABuffers:
    """Symmetric O/LSE sources, control and the compiled kernel for one serial DCP group.

    output()/lse() are the producer's write targets for `rows` rows; merge(rows) returns
    [local_heads, rows, dim]. Allocate before CUDA graph capture (capacity grows only eagerly).
    """

    def __init__(self, heads, record_heads, dim, capacity, device, group):
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.local_heads, self.record_heads, self.dim = heads // self.world, record_heads, dim
        self.capacity = capacity
        self.output_buffer = symm.empty((capacity, record_heads, dim), dtype=torch.bfloat16, device=device)
        self.lse_buffer = symm.empty((capacity, record_heads), dtype=torch.float32, device=device)
        self.control = symm.empty((CONTROL_SIZE,), dtype=torch.int64, device=device)
        handles = [symm.rendezvous(t, group=group) for t in (self.output_buffer, self.lse_buffer, self.control)]
        self.peers = torch.tensor(
            list(zip(handles[0].buffer_ptrs, handles[1].buffer_ptrs)), dtype=torch.int64, device=device
        )
        self.control_peers = torch.tensor(handles[2].buffer_ptrs, dtype=torch.int64, device=device)
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.warps = warps_for(self.world, dim)
        self.kernel = compile_mega_a2a(self.world, self.local_heads, record_heads, self.warps, dim)
        self.control.zero_()
        # Every peer's control must be zeroed before any publication.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def output(self, rows):
        return self.output_buffer[:rows]

    def lse(self, rows):
        return self.lse_buffer[:rows]

    def merge(self, rows, out=None):
        if out is None:
            out = self.output_buffer.new_empty(self.local_heads, rows, self.dim)
        ctas = min(self.sm_count, triton.cdiv(rows * self.local_heads, self.warps))
        with tvm_ffi.use_torch_stream():
            self.kernel(self.peers, self.control_peers, out, Int32(rows), Int32(self.rank), Int32(ctas))
        return out


_buffers = {}
# Replaced buffers stay alive: graphs captured against them keep their symmetric addresses, and an
# eager batch beyond the largest captured one grows the buffers after capture.
_retired = []


def get_mega_a2a_buffers(heads, record_heads, dim, rows, device, group):
    """The group's shared buffers, grown geometrically (eager only) to hold `rows` rows."""
    key = (group, device, heads, record_heads, dim)
    current = _buffers.get(key)
    if current is None or current.capacity < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MegaA2A buffers must be allocated before graph capture")
        if current is not None:
            _retired.append(current)
        capacity = max(16, 1 << (rows - 1).bit_length())
        current = _buffers[key] = MegaA2ABuffers(heads, record_heads, dim, capacity, device, group)
    return current
