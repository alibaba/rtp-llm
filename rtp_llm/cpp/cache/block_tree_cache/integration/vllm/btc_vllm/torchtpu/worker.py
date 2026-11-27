# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""BTC's rank-local PJRT transport for whole TPU KV pages."""

import fcntl
import hashlib
import mmap
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from btc_vllm.torchtpu.layout import BtcHostRows
from btc_vllm.torchtpu.rank_transfers import PageSegments, RankResult, RankTransfers
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class BtcHostGeometry:
    engine_id: str
    num_ranks: int
    rank_bytes: int
    row_bytes: int
    num_slots: int

    @classmethod
    def from_config(
        cls, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
    ) -> "BtcHostGeometry":
        from vllm_torchtpu.block_major_pool import unified_block_major_enabled
        from btc_vllm.torchtpu.worker_layout import (
            rank_page_bytes,
        )

        if unified_block_major_enabled(vllm_config):
            raise ValueError(
                "BTC's single-host worker requires the separate native TPU pools; "
                "set VLLM_TPU_BLOCK_MAJOR_KV=0"
            )

        parallel = vllm_config.parallel_config
        ranks = parallel.tensor_parallel_size
        if (
            ranks != parallel.local_world_size
            or ranks != parallel.world_size
            or parallel.nnodes_within_dp != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError(
                "BTC TPU Host storage requires node-local TP within each DP engine"
            )
        groups = kv_cache_config.kv_cache_groups
        if (
            not groups
            or not kv_cache_config.kv_cache_tensors
            or kv_cache_config.num_blocks <= 0
        ):
            raise ValueError("BTC TPU transfer requires allocated, paged KV groups")

        rank_bytes = rank_page_bytes(vllm_config, kv_cache_config)
        if rank_bytes <= 0:
            raise ValueError("BTC TPU requires one whole page per rank")
        size = ranks * rank_bytes
        stride = ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None and transfer.engine_id is not None
        budget = transfer.get_from_extra_config("cpu_bytes_to_use", 0)
        if type(budget) is not int or budget < stride:
            raise ValueError(
                f"BTC cpu_bytes_to_use must hold at least one {stride}-byte Host slot"
            )
        identity = f"{os.getuid()}/{parallel.data_parallel_index}/{transfer.engine_id}"
        name = "btc_" + hashlib.sha256(identity.encode()).hexdigest()
        return cls(name, ranks, rank_bytes, stride, budget // stride)


class BtcTpuHostRegion(SharedOffloadRegion):
    """Keep the shared mapping after its engine-private name is detached."""

    def backing_identity(self) -> tuple[int, int, int]:
        assert self.fd is not None
        backing = os.fstat(self.fd)
        return backing.st_dev, backing.st_ino, backing.st_size

    def detach_name(self) -> bool:
        assert self.fd is not None
        # Serialize the scheduler's detach with the creator's failure cleanup
        # on this inode. After unlink, an engine may reuse the pathname.
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            backing = os.fstat(self.fd)
            if not backing.st_nlink:
                return True
            try:
                named = os.stat(self.mmap_path, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if (named.st_dev, named.st_ino) != (backing.st_dev, backing.st_ino):
                return False
            os.unlink(self.mmap_path)
            return True
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def cleanup(self) -> None:
        if self._creator and self.fd is not None:
            try:
                self.detach_name()
            except OSError:
                logger.warning("Failed to detach BTC Host rendezvous %s", self.mmap_path,
                               exc_info=True)
            # The base class closes fd before unlinking and cannot check the inode.
            self._creator = False
        super().cleanup()


def open_tpu_host_region(geometry: BtcHostGeometry) -> BtcTpuHostRegion:
    # Every TP process and the scheduler attach the same engine-private mapping.
    # Only the workers DMA-map it. The scheduler detaches its name at handshake.
    return BtcTpuHostRegion(
        engine_id=geometry.engine_id,
        num_chunks=geometry.num_slots,
        rank=None,
        kv_bytes_per_chunk=geometry.row_bytes,
        cpu_page_size=geometry.rank_bytes,
    )


class BtcTpuWorker:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        geometry: BtcHostGeometry,
    ) -> None:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        self._config = kv_cache_config
        self._geometry = geometry
        self._rank = get_tensor_model_parallel_rank()
        if not 0 <= self._rank < geometry.num_ranks:
            raise ValueError(f"TPU rank {self._rank} is outside the Host row layout")
        self._host_rows = BtcHostRows(
            geometry.num_slots, geometry.row_bytes,
            (geometry.rank_bytes,) * len(kv_cache_config.kv_cache_groups),
        )
        self._region: BtcTpuHostRegion | None = None
        self._jobs: RankTransfers | None = None
        self._tensors: list[torch.Tensor] = []
        # Pairs of (offset within a rank's wire row, bytes per HBM block).
        self._regions: list[tuple[int, int]] = []

    @property
    def rank(self) -> int:
        return self._rank

    def host_backing_identity(self) -> tuple[int, int, int]:
        assert self._region is not None, "BTC TPU buffers have not been registered"
        return self._region.backing_identity()

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        from _btc_tpu_transfer import RankTransfer

        from btc_vllm.torchtpu.worker_layout import (
            resolve_rank_page,
        )
        from vllm_torchtpu.utils import synchronize_tensors

        if self._jobs is not None:
            raise RuntimeError("BTC TPU buffers were already registered")
        tensors = []
        regions = []
        offset = 0
        for tensor, page_bytes in resolve_rank_page(self._config, kv_caches):
            tensors.append(tensor)
            regions.append((offset, page_bytes))
            offset += page_bytes
        if offset != self._geometry.rank_bytes:
            raise ValueError(
                f"BTC TPU rank layout is {offset} bytes, "
                f"expected {self._geometry.rank_bytes}"
            )
        synchronize_tensors(tensors, wait=True)
        region = open_tpu_host_region(self._geometry)
        try:
            runtime = RankTransfer(
                tensors, region.base_tensor.view(torch.uint8), self._config.num_blocks,
                self._geometry.num_slots, len(self._config.kv_cache_groups), self._rank,
            )
        except BaseException:
            region.cleanup()
            raise
        expected_sizes = [size * self._config.num_blocks for _, size in regions]
        if runtime.buffer_sizes() != expected_sizes:
            runtime.close()
            region.cleanup()
            raise ValueError(
                "PJRT physical buffer sizes disagree with the BTC wire layout"
            )
        self._region = region
        self._jobs = RankTransfers(runtime, self._rank)
        self._regions = regions
        self._tensors = tensors

    def submit(
        self, job_id: int, load: bool, pages: Sequence[tuple[int, int, int]]
    ) -> None:
        """Submit (cache group, HBM block, BTC host slot) pages on this rank."""
        from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

        if self._jobs is None:
            raise RuntimeError("BTC TPU transfer requires registered buffers")
        rows: list[PageSegments] = []
        for group, block, slot in pages:
            if not 0 < block < self._config.num_blocks:
                raise ValueError(f"BTC job {job_id} has invalid HBM page {block}")
            start, _ = self._host_rows.bounds(group, slot)
            host = start + self._rank * self._geometry.rank_bytes
            spans = [
                (buffer_idx, block * size, host + region_offset, size)
                for buffer_idx, (region_offset, size) in enumerate(self._regions)
            ]
            rows.append((group, block, slot, spans))
        # Materialize deferred KV producers before taking the source snapshot.
        # The hardware barrier also orders reuse behind already-enqueued readers.
        synchronize_tensors(self._tensors, wait=True)
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
        if self._region is not None:
            self._region.cleanup()
            self._region = None
        self._tensors.clear()
