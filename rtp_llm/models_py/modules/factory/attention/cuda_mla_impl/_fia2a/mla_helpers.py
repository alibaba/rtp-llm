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

import cutlass
import cutlass.cute as cute
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
