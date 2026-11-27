# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native BTC DMA pages and the distributed startup layout handshake."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from vllm.config import ParallelConfig


def layout_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class BtcRank:
    global_rank: int
    node: int
    local_rank: int


@dataclass(frozen=True)
class BtcTopology:
    pp_size: int
    pcp_size: int
    tp_size: int
    local_world_size: int
    num_nodes: int

    @classmethod
    def from_config(cls, parallel: ParallelConfig) -> BtcTopology:
        topology = cls(
            parallel.pipeline_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.tensor_parallel_size,
            parallel.local_world_size,
            parallel.nnodes_within_dp,
        )
        if (
            parallel.decode_context_parallel_size != 1
            or topology.num_ranks != parallel.world_size
            or topology.num_ranks != topology.num_nodes * topology.local_world_size
        ):
            raise ValueError("BTC requires a complete PP/PCP/TP world with DCP1")
        return topology

    @property
    def num_ranks(self) -> int:
        return self.pp_size * self.pcp_size * self.tp_size

    def rank(self, pp: int, pcp: int, tp: int, node: int) -> BtcRank:
        if not (
            0 <= pp < self.pp_size
            and 0 <= pcp < self.pcp_size
            and 0 <= tp < self.tp_size
        ):
            raise ValueError(
                f"BTC worker coordinates outside its topology: {pp, pcp, tp}"
            )
        global_rank = (pp * self.pcp_size + pcp) * self.tp_size + tp
        rank_node, local_rank = divmod(global_rank, self.local_world_size)
        if rank_node != node:
            raise ValueError(f"BTC worker rank {global_rank} is outside node {node}")
        return BtcRank(global_rank, node, local_rank)

    def worker_rank(self, node_rank: int) -> BtcRank:
        from vllm.distributed.parallel_state import (
            get_pcp_group,
            get_pp_group,
            get_tp_group,
        )

        return self.rank(
            get_pp_group().rank_in_group,
            get_pcp_group().rank_in_group,
            get_tp_group().rank_in_group,
            node_rank,
        )


@dataclass(frozen=True)
class BtcRankLayout:
    rank: BtcRank
    fingerprint: str
    group_bytes: tuple[int, ...]


@dataclass(frozen=True)
class BtcHostRows:
    """Address the payload each group occupies in a rank-local Host slot."""

    num_slots: int
    row_bytes: int
    group_bytes: tuple[int, ...]

    def bounds(self, group: int, slot: int) -> tuple[int, int]:
        if type(group) is not int or not 0 <= group < len(self.group_bytes):
            raise ValueError(f"Invalid BTC Host group {group!r}")
        if type(slot) is not int or not 0 <= slot < self.num_slots:
            raise ValueError(f"Invalid BTC Host slot {slot!r}")
        first = slot * self.row_bytes
        return first, first + self.group_bytes[group]


@dataclass(frozen=True)
class BtcStorageLayout:
    ranks: tuple[BtcRankLayout, ...]
    fingerprint: str

    @classmethod
    def from_ranks(
        cls,
        topology: BtcTopology,
        descriptions: Iterable[BtcRankLayout],
        group_count: int,
    ) -> BtcStorageLayout:
        ranks = tuple(sorted(descriptions, key=lambda row: row.rank.global_rank))
        if len(ranks) != topology.num_ranks or group_count <= 0:
            raise ValueError("BTC layout handshake must cover every worker and group")
        for index, row in enumerate(ranks):
            node, local_rank = divmod(index, topology.local_world_size)
            if (
                row.rank != BtcRank(index, node, local_rank)
                or len(row.group_bytes) != group_count
                or not row.fingerprint
                or any(type(size) is not int or size < 0 for size in row.group_bytes)
            ):
                raise ValueError(f"Invalid BTC layout handshake at rank {index}: {row}")
        if any(not any(group) for group in zip(*(row.group_bytes for row in ranks))):
            raise ValueError("BTC cannot publish a group with no owning worker")
        fingerprint = layout_fingerprint(
            ("btc-packed-pages-v1", asdict(topology), [asdict(row) for row in ranks])
        )
        return cls(ranks, fingerprint)

    def payload_bytes_by_rank(self, group: int) -> tuple[int, ...]:
        return tuple(row.group_bytes[group] for row in self.ranks)

    def participants(self, group: int) -> tuple[int, ...]:
        return tuple(
            row.rank.global_rank for row in self.ranks if row.group_bytes[group]
        )


@dataclass(frozen=True)
class BtcDeviceSlice:
    storage_index: int
    block_stride_bytes: int
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class BtcPageLayout:
    buffers: tuple[Tensor, ...]
    groups: tuple[tuple[BtcDeviceSlice, ...], ...]
    fingerprint: str

    @property
    def buffer_sizes(self) -> list[int]:
        return [buffer.nbytes for buffer in self.buffers]

    def payload_bytes(self, group: int) -> int:
        return sum(span.size_bytes for span in self.groups[group])

    def describe(self, rank: BtcRank) -> BtcRankLayout:
        return BtcRankLayout(
            rank,
            self.fingerprint,
            tuple(map(self.payload_bytes, range(len(self.groups)))),
        )

    def spans(
        self, group: int, block: int, host_offset: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        result = []
        for span in self.groups[group]:
            result.append(
                (
                    span.storage_index,
                    block * span.block_stride_bytes + span.offset_bytes,
                    host_offset,
                    span.size_bytes,
                )
            )
            host_offset += span.size_bytes
        return tuple(result)
