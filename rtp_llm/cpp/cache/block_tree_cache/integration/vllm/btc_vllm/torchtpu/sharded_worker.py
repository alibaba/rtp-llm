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
"""Rank-local native transfers for sharded TorchTPU caches."""

from __future__ import annotations

import mmap
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch

from btc_vllm.torchtpu.layout import (
    BtcHostRows,
    BtcPageLayout,
    BtcRankLayout,
    BtcTopology,
)
from btc_vllm.torchtpu.rank_transfers import PageSegments, RankResult, RankTransfers
from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class ShardedBtcWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        cache_config: KVCacheConfig,
        topology: BtcTopology,
    ) -> None:
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        host_budget = transfer.get_from_extra_config("cpu_bytes_to_use", 0)
        if type(host_budget) is not int or host_budget <= 0:
            raise ValueError("Sharded BTC requires a positive cpu_bytes_to_use per node")
        self._rank = topology.worker_rank(
            vllm_config.parallel_config.node_rank_within_dp
        )
        self._budget = host_budget // topology.local_world_size
        self._num_blocks = cache_config.num_blocks
        self._layout: BtcPageLayout | None = None
        self._jobs: RankTransfers | None = None
        self.region: mmap.mmap | None = None
        self.host_fd: int | None = None
        self._host_rows: BtcHostRows | None = None

    @property
    def rank(self) -> int:
        return self._rank.global_rank

    @property
    def host_rows(self) -> BtcHostRows:
        assert self._host_rows is not None, "BTC Host rows are not registered"
        return self._host_rows

    @property
    def row_bytes(self) -> int:
        return self.host_rows.row_bytes

    @property
    def num_slots(self) -> int:
        return self.host_rows.num_slots

    def register_layout(self, layout: BtcPageLayout) -> None:
        from _btc_tpu_transfer import RankTransfer

        assert self._layout is None, "BTC buffers are already registered"
        description = layout.describe(self._rank)
        max_payload = max(description.group_bytes)
        row_bytes = (max_payload + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        num_slots = self._budget // row_bytes if row_bytes else 0
        if num_slots <= 0:
            raise ValueError(
                f"BTC rank {self.rank} needs {row_bytes} host bytes per slot, "
                f"but received a {self._budget}-byte budget"
            )
        fd = os.memfd_create(f"btc-host-rank-{self.rank}")
        region = None
        runtime = None
        try:
            size = num_slots * row_bytes
            os.ftruncate(fd, size)
            region = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            synchronize_tensors(layout.buffers, wait=True)
            runtime = RankTransfer(
                layout.buffers,
                torch.frombuffer(region, dtype=torch.uint8),
                self._num_blocks,
                num_slots,
                len(layout.groups),
                self.rank,
            )
            if runtime.buffer_sizes() != layout.buffer_sizes:
                raise ValueError(
                    "BTC device carrier size differs from its layout"
                )
        except BaseException:
            if runtime is not None:
                runtime.close()
            if region is not None:
                region.close()
            os.close(fd)
            raise
        self.region = region
        self.host_fd = fd
        self._host_rows = BtcHostRows(num_slots, row_bytes, description.group_bytes)
        self._layout = layout
        self._jobs = RankTransfers(runtime, self.rank)

    def describe(self) -> BtcRankLayout:
        assert self._layout is not None, "BTC layout registration is incomplete"
        return self._layout.describe(self._rank)

    def host_view(self, group: int, slot: int) -> memoryview:
        assert self.region is not None
        start, end = self.host_rows.bounds(group, slot)
        return memoryview(self.region)[start:end]

    def submit(
        self, job_id: int, load: bool, pages: Sequence[tuple[int, int, int]]
    ) -> None:
        layout = self._layout
        assert layout is not None and self._jobs is not None
        rows: list[PageSegments] = []
        for group, block, slot in pages:
            assert 0 < block < self._num_blocks, f"Invalid BTC device block {block}"
            start, _ = self.host_rows.bounds(group, slot)
            rows.append((group, block, slot, layout.spans(group, block, start)))
        if any(spans for _, _, _, spans in rows):
            synchronize_tensors(layout.buffers, wait=True)
            synchronize_device()
        self._jobs.submit(job_id, load, rows)

    def wait(self, job_ids: Iterable[int]) -> bool:
        assert self._jobs is not None
        return self._jobs.wait(job_ids)

    def get_finished(self) -> list[tuple[int, RankResult]]:
        assert self._jobs is not None
        return self._jobs.get_finished()

    def transfer_stats(self) -> tuple[int, int, int]:
        assert self._jobs is not None
        return self._jobs.stats()

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.close()
            self._jobs = None
        if self.region is not None:
            self.region.close()
            self.region = None
            assert self.host_fd is not None
            os.close(self.host_fd)
            self.host_fd = None
