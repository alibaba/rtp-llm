# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from dataclasses import dataclass
from math import gcd
from types import SimpleNamespace
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

from .ops import CONTROL_SIZE, COUNT, EPOCH, READY


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        split_kv_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler parameters prepared for MLA static tile scheduler.

        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param problem_shape_b: The shape of the problem
        :type problem_shape_b: cute.Int32
        :param problem_shape_s: The shape of the problem in sequence length Q dimension
        :type problem_shape_s: cute.Int32
        :param cluster_shape_mnk: The shape of the cluster
        :type cluster_shape_mnk: cute.Shape
        :param split_kv: The scalar factor for split KV
        """
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor(
                split_kv, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        problem_shape_s = cutlass.new_from_mlir_values(
            self.problem_shape_s, (values[1],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        problem_shape_b_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_b_fdd, (values[3],)
        )
        problem_shape_s_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_s_fdd, (values[4],)
        )
        split_kv_fdd = cutlass.new_from_mlir_values(self.split_kv_fdd, (values[5],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, problem_shape_s, cluster_shape_mnk, split_kv
    )


class WorkTileInfo:
    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class MLAStaticTileScheduler:
    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler for MLA split kv kernel.
        Based on `is_persistent`, it provides 2 modes for use:
        - Persistent mode: Launch fixed blocks and reschedule the data blocks.
        - Non-persistent mode: Launch dynamic blocks and exit when the current work is done.

        :param params: The static tile scheduler parameters
        :type params: MLAStaticTileSchedulerParams
        :param current_work_linear_idx: The linear index of the current work
        :type current_work_linear_idx: cutlass.Int32
        :param blk_coord: The coordinate of the current work
        :type blk_coord: cute.Coord
        :param grid_shape: The shape of the grid
        :type grid_shape: cute.Shape
        :param is_valid: Whether the current work is valid
        :type is_valid: bool
        """
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    params.problem_shape_s,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            # Used for persistent scheduling
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b * params.problem_shape_s,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            current_work_cluster_batch, cluster_idx = (
                self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
                self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
            )
            current_work_s_batch, s_idx = divmod(
                current_work_cluster_batch, self.params.problem_shape_s_fdd
            )
            current_work_b_batch, b_idx = divmod(
                current_work_s_batch, self.params.problem_shape_b_fdd
            )
            _, split_kv_idx = divmod(current_work_b_batch, self.params.split_kv_fdd)

            blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        else:
            b_idx, s_idx = divmod(self.blk_coord[1], self.params.problem_shape_s_fdd)
            blk_coord = (self.blk_coord[0], s_idx, b_idx, self.blk_coord[2])

        return WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 13
        new_params = cutlass.new_from_mlir_values(self.params, values[0:6])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[6]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[7:10])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[10:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


LOG2_E = 1.4426950408889634074




def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_q_tile_layout(
    num_heads: int, seq_len_q: int, m_tile: int = 128
) -> tuple[int, int, int]:
    """Return ``(total_rows, num_tiles, tail_rows)`` for flat query packing.

    Query-token and head modes are one affine row space ordered as
    ``flat_row = q_token * num_heads + q_head``.  Consecutive M tiles may
    therefore cross token boundaries; only the final tile can be partial.
    ``tail_rows`` is always in ``[1, m_tile]`` and equals ``m_tile`` when the
    flattened row count exactly fills the final tile.

    This host-side helper is shared by launch/workspace selection and both
    kernel variants so their split-KV geometry cannot drift apart.
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if seq_len_q <= 0:
        raise ValueError(f"seq_len_q must be positive, got {seq_len_q}")
    if m_tile <= 0:
        raise ValueError(f"m_tile must be positive, got {m_tile}")
    if num_heads > m_tile:
        raise ValueError(
            f"num_heads ({num_heads}) must not exceed the MMA M tile ({m_tile})"
        )

    total_rows = num_heads * seq_len_q
    num_tiles = ceil_div(total_rows, m_tile)
    tail_rows = total_rows - (num_tiles - 1) * m_tile
    return total_rows, num_tiles, tail_rows


# Packed query/head output routing. Each two-CTA cluster owns 128 rows;
# phase repetition keeps the route table bounded for long speculative Q.
@dataclass(frozen=True)
class OutputRoutes:
    heads: int
    queries: int
    world: int
    phase_count: int
    queries_per_period: int
    repeating: bool
    routes: tuple
    groups: tuple
    boxes: tuple
    tma_3d_eligible: bool


def make_output_routes(heads: int, queries: int, world: int) -> OutputRoutes:
    if not 1 <= heads <= 128 or queries <= 0 or world <= 0 or heads % world:
        raise ValueError("Packed MLA requires 1..128 heads, positive Q/world, and equal head shards")
    period = heads // gcd(heads, 128)
    query_tiles = (queries * heads + 127) // 128
    phases = min(period, query_tiles)
    local_heads = heads // world
    routes, boxes, groups = [], [], []
    for phase in range(phases):
        for cta in range(2):
            group = []
            row = 0
            while row < 64:
                flat = phase * 128 + cta * 64 + row
                query_delta, head = divmod(flat, heads)
                destination, local_head = divmod(head, local_heads)
                count = min(64 - row, heads - head, local_heads - local_head)
                box = (destination, count)
                if box not in boxes:
                    boxes.append(box)
                group.append(len(routes))
                routes.append((phase, cta, row, query_delta, destination,
                               local_head, count, boxes.index(box)))
                row += count
            groups.append(tuple(group))
    # Both BF16 and FP32 output use 256 bytes per staged row. SW128
    # requires each route's shared-memory base to remain 1024-byte aligned.
    # Ineligible shards use vector P2P stores instead of lying about alignment.
    eligible = all(route[2] % 4 == 0 and route[6] % 4 == 0 for route in routes)
    return OutputRoutes(heads, queries, world, phases, period * 128 // heads,
                        query_tiles > period, tuple(routes), tuple(groups),
                        tuple(boxes), eligible)


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    # CuTe 4.4.2 has no arch.globaltimer wrapper. This register counts ns.
    return Int64(llvm.inline_asm(T.i64(), [], 'mov.u64 $0, %globaltimer;', '=l',
                                has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def timeout_trap(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'trap;', '', has_side_effects=True,
                    asm_dialect=0, loc=loc, ip=ip)


# Peer publication uses the FIA2A control layout (one int64 array per rank in
# symmetric memory): completed epoch, finished-CTA count and ready[src], where
# a receiver's ready[src] is written only by rank src with that call's epoch.


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


@cute.jit
def publish_peer_outputs(control_peers: cute.Tensor, source_rank: cutlass.Int32,
                         world: cutlass.Constexpr, tidx: cutlass.Int32):
    """Exchange completion with every peer once all CTAs finished their stores.

    Call from all threads of every CTA after its peer O/LSE stores completed
    (TMA stores waited by their issuer) and a CTA barrier ordered every
    thread's remote stores before this point. The last arriving CTA of the
    persistent grid publishes this rank's completion, then waits until every
    peer has published: when the kernel exits, all records this rank will
    read have arrived, so the stream-ordered consumer needs no synchronization.
    Each rank publishes before it waits, so the exchange cannot deadlock.
    """
    if tidx == 0:
        control = cute.make_tensor(
            cute.make_ptr(cutlass.Int64, control_peers[source_rank], cute.AddressSpace.gmem, assumed_align=8),
            cute.make_layout((CONTROL_SIZE,)))
        # Releases this CTA's barrier-ordered stores; the last arrival also
        # acquires every earlier CTA's release through the RMW sequence.
        finished = cute.arch.atomic_add((control.iterator + COUNT).llvm_ptr, cutlass.Int64(1),
                                        sem="acq_rel", scope="gpu")
        grid = cute.arch.grid_dim()
        if finished == cutlass.Int64(grid[0] * grid[1] * grid[2] - 1):
            epoch = control[EPOCH] + 1
            # Fence + strong writes form one system release for all peers.
            cute.arch.fence_acq_rel_sys()
            for destination in cutlass.range_constexpr(world):
                ready = cute.make_ptr(cutlass.Int64, control_peers[destination],
                                      cute.AddressSpace.gmem, assumed_align=8)
                cute.arch.store((ready + READY + source_rank).llvm_ptr, epoch,
                                sem="relaxed", scope="sys")
            for source in cutlass.range_constexpr(world):
                wait_ready(control.iterator + READY + source, epoch)
            # Nothing reads these again until this kernel has exited.
            control[COUNT] = cutlass.Int64(0)
            control[EPOCH] = epoch


# Shared compatibility and tile operations for the BF16 and FP8 producers.
# Compat shim: setmaxregister_{decrease,increase} added in cutlass-dsl 4.4;
# older versions only have the deprecated warpgroup_reg_{dealloc,alloc}.
_setmaxregister_decrease = getattr(
    cute.arch,
    "setmaxregister_decrease",
    getattr(cute.arch, "warpgroup_reg_dealloc", None),
)
_setmaxregister_increase = getattr(
    cute.arch,
    "setmaxregister_increase",
    getattr(cute.arch, "warpgroup_reg_alloc", None),
)

# Compat shim: get_max_tmem_alloc_cols added in cutlass-dsl 4.4;
# older versions don't have it, so we provide a fallback implementation.
_TMEM_MAX_ALLOC_COLUMNS_MAP = {"sm_100": 512, "sm_103": 512, "sm_120": 512}


def _get_max_tmem_alloc_cols(compute_capability: str) -> int:
    if hasattr(cute.arch, "get_max_tmem_alloc_cols"):
        return cute.arch.get_max_tmem_alloc_cols(compute_capability)
    if compute_capability not in _TMEM_MAX_ALLOC_COLUMNS_MAP:
        raise ValueError(f"Unsupported compute capability: {compute_capability}")
    return _TMEM_MAX_ALLOC_COLUMNS_MAP[compute_capability]


@cute.jit
def make_paged_tiled_tma_atom(
    kernel: cutlass.Constexpr,
    tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    is_k_load: bool,
):
    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    page_tile_size = (
        min(kernel.page_size, cta_mn)
        if is_k_load
        else min(kernel.page_size, mma_tiler[1])
    )
    cta_v_map = cute.zipped_divide(
        cta_v_map,
        (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
    )
    cta_v_map = cute.select(cta_v_map, mode=[0])
    from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

    res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
        gmem.value,
        smem_layout.value,
        cta_v_map,
        tma_load_op._to_ir(),
        num_multicast=1,
    )
    return cute.CopyAtom(
        tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
    ), res[1]


@cute.jit
def get_k_tile_count(
    kernel: cutlass.Constexpr,
    split_kv: cutlass.Int32,
    cache_seqs: cute.Tensor,
    blk_coord: cute.Coord,
) -> tuple[
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
]:
    """Get K split and request-local query metadata for one work item.

    :param split_kv: Split_kv value
    :type split_kv: cutlass.Int32
    :param cache_seqs: Cache sequence lengths tensor
    :type cache_seqs: cute.Tensor
    :param blk_coord: Block coordinate
    :type blk_coord: cute.Coord
    :return: K range, effective split count, and request-local Q metadata
    """
    K = cache_seqs[blk_coord[2]]

    k_tile_total = cute.ceil_div(K, kernel.mma_qk_tiler[1])
    k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
    k_index = blk_coord[3] * k_tile_per_cta
    k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
    q_begin = cutlass.Int32(0)
    q_len = cutlass.Int32(kernel.seq_len_q)
    valid_q_rows = kernel.get_valid_q_rows(blk_coord[1])
    return (
        k_index,
        k_tile_count,
        split_kv,
        q_begin,
        q_len,
        valid_q_rows,
    )


@cute.jit
def correction(
    kernel: cutlass.Constexpr,
    common_params: SimpleNamespace,
    epilogue_params: SimpleNamespace,
    k_tile_count: cutlass.Int32,
    p_cor_consumer_state: pipeline.PipelineState,
    mma_o_consumer_state: pipeline.PipelineState,
) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
    """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

    :param common_params: The common parameters
    :type common_params: SimpleNamespace
    :param epilogue_params: The epilogue parameters
    :type epilogue_params: SimpleNamespace
    :param k_index: The index of the k-tile
    :type k_index: cutlass.Int32
    :param k_tile_count: The number of k-tiles
    :type k_tile_count: cutlass.Int32
    :param p_cor_consumer_state: The P correction consumer state
    :type p_cor_consumer_state: pipeline.PipelineState
    :param mma_o_consumer_state: The MMA o consumer state
    :type mma_o_consumer_state: pipeline.PipelineState

    :return: The P correction consumer state, and the MMA o consumer state
    :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
    """

    k_tile_count_init = k_tile_count
    while k_tile_count > 0:
        p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction = (
            kernel.get_correction_factor(common_params, p_cor_consumer_state)
        )
        if k_tile_count_init != k_tile_count:
            mma_o_consumer_state = kernel.rescale(
                common_params,
                mma_o_consumer_state,
                correction_factor,
                no_correction,
            )
        k_tile_count = k_tile_count - 1
        if k_tile_count == 0:
            mma_o_consumer_state = kernel.epilogue(
                common_params,
                epilogue_params,
                mma_o_consumer_state,
                row_sum,
                row_max,
            )

    return p_cor_consumer_state, mma_o_consumer_state


@cute.jit
def _tmem_load_partition(
    kernel: cutlass.Constexpr, common_params: SimpleNamespace, tiled_mma_pv: cute.TiledMma, iter_n: int
) -> tuple[
    cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
]:
    """Tensor memory load partition for rescale and epilogue.

    :param common_params: The common parameters
    :type common_params: SimpleNamespace
    :param tiled_mma_pv: The tiled mma pv
    :type tiled_mma_pv: cute.TiledMma
    :param iter_n: The iteration number
    :type iter_n: int

    :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
    :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
    """

    tOtO_shape = tiled_mma_pv.partition_shape_C(
        cute.select(kernel.mma_pv_tiler, mode=[0, 1])
    )
    tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
    tOtO_layout = cute.append(
        tOtO.layout,
        cute.make_layout(
            common_params.L // kernel.mma_pv_tiler[1],
            stride=kernel.mma_pv_tiler[1] // kernel.warps_in_n,
        ),
    )
    tOtO = cute.make_tensor(
        common_params.tmem_ptr + kernel.tmem_o_offset, tOtO_layout
    )
    tOtO = tOtO[None, None, None, iter_n]

    tAcc = tOtO[(None, None), 0, 0]

    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), kernel.acc_dtype
    )
    tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
    tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
        common_params.tidx % (kernel.num_compute_warps * kernel.threads_per_warp)
    )

    cta_pv_tiler = (
        kernel.mma_pv_tiler[0] // kernel.cluster_shape_mnk[0],
        kernel.mma_pv_tiler[1],
        kernel.mma_pv_tiler[2],
    )
    # Flatten divide and partition global tensors for O
    cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

    gO = None
    gO = cute.local_tile(
        common_params.mO,
        cta_pv_tiler_mn,
        (
            common_params.blk_coord[0],
            iter_n,
            common_params.blk_coord[1],
            common_params.blk_coord[2],
        ),
    )
    cO = cute.local_tile(
        cute.make_identity_tensor(common_params.mO.shape),
        cta_pv_tiler_mn,
        (
            common_params.blk_coord[0],
            iter_n,
            common_params.blk_coord[1],
            common_params.blk_coord[2],
        ),
    )
    tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
    tTR_gO = tmem_load_thr_copy.partition_D(gO)
    tTR_cO = tmem_load_thr_copy.partition_D(cO)
    tTR_rAcc = cute.make_fragment_like(tTR_gO, kernel.acc_dtype)
    return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc  # type: ignore[return-value]


def get_correction_factor(
    kernel,
    common_params: SimpleNamespace,
    p_cor_consumer_state: pipeline.PipelineState,
) -> tuple[
    pipeline.PipelineState,
    cutlass.Float32,
    cutlass.Float32,
    cutlass.Float32,
    cutlass.Int32,
]:
    """Get the correction factor from the P correction consumer state.

    :param common_params: The common parameters
    :type common_params: SimpleNamespace
    :param p_cor_consumer_state: The P correction consumer state
    :type p_cor_consumer_state: pipeline.PipelineState

    :return: The P correction consumer state, the row_sum, the row_max, and the correction factor
    :rtype: tuple[pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32]
    """
    common_params.p_cor_pipeline.consumer_wait(p_cor_consumer_state)
    tidx = common_params.tidx % (kernel.num_compute_warps * kernel.threads_per_warp)
    # load correction factor
    _, tAcc, _, _, _, _ = kernel._tmem_load_partition(
        common_params, common_params.tiled_mma_pv, 0
    )
    corr_layout = cute.make_layout(
        (tAcc.shape[0], (4, tAcc.shape[1][1]), kernel.p_cor_stage),
        stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
    )
    tCor = cute.make_tensor(
        common_params.tmem_ptr + kernel.correction_factor_offset, corr_layout
    )
    cCor = cute.make_identity_tensor(tCor.shape)
    corr_tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), kernel.acc_dtype
    )
    corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_load_atom, tCor)
    corr_tmem_load_thr_copy = corr_tmem_load_tiled_copy.get_slice(tidx)
    tCor_for_copy = corr_tmem_load_thr_copy.partition_S(tCor)
    cCor_for_copy = corr_tmem_load_thr_copy.partition_D(cCor)
    rCor = cute.make_fragment_like(
        cCor_for_copy[None, None, None, 0], kernel.acc_dtype
    )
    rCor_int = cute.make_tensor(
        cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
    )
    cute.copy(
        corr_tmem_load_tiled_copy,
        tCor_for_copy[None, None, None, p_cor_consumer_state.index],
        rCor,
    )
    row_sum = rCor[0]
    row_max = rCor[1]
    correction_factor = rCor[2]
    no_correction = rCor_int[3]

    common_params.p_cor_pipeline.consumer_release(p_cor_consumer_state)
    p_cor_consumer_state.advance()
    return p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction


@cute.jit
def pagerr_local_bound(kernel: cutlass.Constexpr, query_local_ends, request_index, query_index):
    bound = cutlass.Int32(0)
    if query_index < kernel.seq_len_q:
        bound = query_local_ends[request_index * kernel.seq_len_q + query_index]
    return bound

# Pipeline cooperative groups below take participant counts, not warp IDs.
# Both producers use one issuing thread per MMA/TMA role; their distinct
# warp dispatch and stage counts remain in the BF16/FP8 kernel classes.
def make_and_init_load_qkv_pipeline(
    kernel, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
) -> pipeline.PipelineTmaUmma:
    """Create and initialize the tma load qkv pipeline.

    :param load_qkv_mbar_ptr: The load qkv mbar pointer
    :type load_qkv_mbar_ptr: cute.Tensor
    :param cta_layout_vmnk: The cta layout vmnk
    :type cta_layout_vmnk: tuple[int, int, int]
    :param load_stages: The load stages
    :type load_stages: list[int]
    :param tx_count: The tx count
    :type tx_count: int

    :return: The tma load qkv pipeline
    :rtype: pipeline.PipelineTmaUmma
    """
    load_qkv_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1
    )
    load_qkv_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1
    )
    return pipeline.PipelineTmaUmma.create(
        barrier_storage=load_qkv_mbar_ptr,
        num_stages=load_stages,
        producer_group=load_qkv_producer_group,
        consumer_group=load_qkv_consumer_group,
        tx_count=tx_count,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )


def make_and_init_mma_s_pipeline(
    kernel, mma_s_mbar_ptr, cta_layout_vmnk
) -> pipeline.PipelineUmmaAsync:
    """Create and initialize the mma s pipeline.

    :param mma_s_mbar_ptr: The mma s mbar pointer
    :type mma_s_mbar_ptr: cute.Tensor
    :param cta_layout_vmnk: The cta layout vmnk
    :type cta_layout_vmnk: tuple[int, int, int]

    :return: The mma s pipeline
    :rtype: pipeline.PipelineUmmaAsync
    """

    mma_s_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1
    )
    consumer_thread_size = (
        kernel.threads_per_warp
        * len(kernel.compute_warp_ids)
        * kernel.cluster_shape_mnk[0]
    )
    mma_s_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_thread_size,
    )
    return pipeline.PipelineUmmaAsync.create(
        barrier_storage=mma_s_mbar_ptr,
        num_stages=kernel.mma_s_stage,
        producer_group=mma_s_producer_group,
        consumer_group=mma_s_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )


def make_and_init_p_mma_pipeline(
    kernel, p_mma_mbar_ptr, cta_layout_vmnk
) -> pipeline.PipelineAsyncUmma:
    """Create and initialize the p mma pipeline.

    :param p_mma_mbar_ptr: The p mma mbar pointer
    :type p_mma_mbar_ptr: cute.Tensor
    :param cta_layout_vmnk: The cta layout vmnk
    :type cta_layout_vmnk: tuple[int, int, int]

    :return: The p mma pipeline
    :rtype: pipeline.PipelineAsyncUmma
    """

    producer_thread_size = (
        kernel.threads_per_warp
        * len(kernel.compute_warp_ids)
        * kernel.cluster_shape_mnk[0]
    )
    p_mma_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_thread_size,
    )
    p_mma_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1
    )
    return pipeline.PipelineAsyncUmma.create(
        barrier_storage=p_mma_mbar_ptr,
        num_stages=kernel.p_mma_stage,
        producer_group=p_mma_producer_group,
        consumer_group=p_mma_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )


def make_and_init_p_cor_pipeline(
    kernel, p_cor_mbar_ptr
) -> pipeline.PipelineAsyncUmma:
    """Create and initialize the p correction pipeline.

    :param p_cor_mbar_ptr: The p correction mbar pointer
    :type p_cor_mbar_ptr: cute.Tensor

    :return: The p correction pipeline
    :rtype: pipeline.PipelineAsyncUmma
    """

    producer_thread_size = kernel.threads_per_warp * len(kernel.compute_warp_ids)
    p_cor_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_thread_size,
    )
    p_cor_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_thread_size,
    )
    return pipeline.PipelineAsync.create(
        barrier_storage=p_cor_mbar_ptr,
        num_stages=kernel.p_cor_stage,
        producer_group=p_cor_producer_group,
        consumer_group=p_cor_consumer_group,
        defer_sync=True,
    )


def make_and_init_mma_o_pipeline(
    kernel, mma_o_mbar_ptr, cta_layout_vmnk
) -> pipeline.PipelineUmmaAsync:
    """Create and initialize the mma o pipeline.

    :param mma_o_mbar_ptr: The mma o mbar pointer
    :type mma_o_mbar_ptr: cute.Tensor
    :param cta_layout_vmnk: The cta layout vmnk
    :type cta_layout_vmnk: tuple[int, int, int]

    :return: The mma o pipeline
    :rtype: pipeline.PipelineUmmaAsync
    """

    mma_o_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1
    )
    consumer_thread_size = (
        kernel.threads_per_warp
        * len(kernel.compute_warp_ids)
        * kernel.cluster_shape_mnk[0]
    )
    mma_o_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_thread_size,
    )
    return pipeline.PipelineUmmaAsync.create(
        barrier_storage=mma_o_mbar_ptr,
        num_stages=kernel.mma_o_stage,
        producer_group=mma_o_producer_group,
        consumer_group=mma_o_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )


def _compute_grid(
    batch_size: cutlass.Int32,
    num_q_tiles: cutlass.Int32,
    split_kv: cutlass.Int32,
    cluster_shape_mnk: Tuple[int, int, int],
    max_active_clusters: int,
    is_persistent: bool,
) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
    """Compute grid shape for the output tensor C.

    :param c: The output tensor C
    :type c: cute.Tensor
    :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
    :type cta_tile_shape_mnk: tuple[int, int, int]
    :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
    :type cluster_shape_mn: tuple[int, int]

    :return: Tile scheduler parameters and grid shape.
    :rtype: tuple[MLAStaticTileSchedulerParams, tuple[int, int, int]]
    """
    tile_sched_params = create_mla_static_tile_scheduler_params(
        is_persistent,
        batch_size,
        num_q_tiles,
        cluster_shape_mnk,
        split_kv,
    )
    grid = MLAStaticTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    return tile_sched_params, grid
