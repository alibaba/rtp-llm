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


"""Hardware-independent planning for packed Page-RR MLA."""

from dataclasses import dataclass
from enum import IntEnum
from math import gcd


class Fia2aFusionMode(IntEnum):
    AUTO = 0
    FUSED = 1
    UNFUSED = 2


class Fia2aA2ABackend(IntEnum):
    AUTO = 0
    NCCL = 1
    CUSTOM = 2


def plan_splits(sm_count, batch, queries, heads):
    if min(sm_count, batch, queries, heads) <= 0:
        raise ValueError("MLA launch dimensions must be positive")
    query_ctas = batch * ((queries * heads + 127) // 128) * 2
    return min(32, max(1, sm_count // query_ctas))


def use_custom_a2a(world, rows, heads):
    return (world in (4, 8) and 768 <= rows * heads <= 49152
            and heads % world == 0 and (rows * (heads // world)) % 4 == 0)


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
