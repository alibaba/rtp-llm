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

"""BlockTreeCache ownership of local TPU and Host pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from btc_vllm.torchtpu.rank_transfers import RankResult
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from block_tree_cache import Operation, Work


@dataclass
class BtcTpuMetadata(KVConnectorMetadata):
    jobs: dict[Operation, Work]


@dataclass(frozen=True)
class BtcHostHandshakeMetadata(KVConnectorHandshakeMetadata):
    rank: int
    backing: tuple[int, int, int]


@dataclass
class BtcRankReceipts(KVConnectorWorkerMetadata):
    jobs: dict[Operation, dict[int, RankResult]]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "BtcRankReceipts":
        assert isinstance(other, BtcRankReceipts)
        merged = {jid: ranks.copy() for jid, ranks in self.jobs.items()}
        for jid, ranks in other.jobs.items():
            seen = merged.setdefault(jid, {})
            for rank, result in ranks.items():
                previous = seen.get(rank)
                seen[rank] = result if previous is None else RankResult(
                    ok=previous.ok and result.ok, safe=previous.safe or result.safe,
                )
        return BtcRankReceipts(merged)


class BtcTpuConnector(KVConnectorBase_V1, SupportsHMA):
    """A single scheduler owns the tree; each worker moves its TP rank's bytes."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self._kv_transfer_config.kv_role != "kv_both":
            raise ValueError("BtcTpuConnector requires kv_both for stores and loads")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("BtcTpuConnector requires prefix caching")
        if kv_cache_config.hisparse_host_num_blocks is not None:
            raise ValueError("BTC TPU requires exclusive Host page ownership")
        self._rank_ops: dict[int, Operation] = {}
        self._init_btc(vllm_config, role, kv_cache_config)

    def _init_btc(self, vllm_config, role, kv_cache_config):
        if vllm_config.speculative_config is not None:
            raise ValueError("BTC TPU does not persist speculative checkpoint offsets")
        from btc_vllm.torchtpu.worker import BtcHostGeometry

        geometry = BtcHostGeometry.from_config(vllm_config, kv_cache_config)
        if role is KVConnectorRole.SCHEDULER:
            from btc_vllm.scheduler import BtcScheduler

            self.scheduler = BtcScheduler(vllm_config, geometry)
        else:
            from btc_vllm.torchtpu.worker import BtcTpuWorker

            self.worker = BtcTpuWorker(vllm_config, kv_cache_config, geometry)

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config):
        # A sleep cycle replaces the PJRT storage addresses registered with BTC.
        return False

    @property
    def requires_kv_delivery(self):
        return False

    def bind_kv_cache_manager(self, kv_cache_manager):
        self.scheduler.bind(kv_cache_manager)

    def try_joint_prefix_match(self, request, max_cache_hit_length):
        return self.scheduler.match(request, max_cache_hit_length)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.scheduler.external(request)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.scheduler.allocated(request, blocks, num_external_tokens)

    def publish_computed_gpu_blocks(self, request, num_committed_tokens):
        self.scheduler.publish(request, num_committed_tokens)

    def build_connector_meta(self, scheduler_output):
        return BtcTpuMetadata(self.scheduler.dispatch(scheduler_output))

    def prepare_scheduler_output(self, output, protected):
        metadata = None if output is None else output.kv_connector_worker_meta
        return self.apply_rank_receipts(output, metadata, protected)

    def apply_rank_receipts(self, output: KVConnectorOutput | None, metadata, protected):
        assert metadata is None or isinstance(metadata, BtcRankReceipts)
        receipts = {} if metadata is None else metadata.jobs
        finished, failed = self.scheduler.settle(receipts, protected)
        if not finished and not failed:
            return output
        if output is None:
            output = KVConnectorOutput()
        if finished:
            output.finished_recving = (output.finished_recving or set()) | finished
        output.failed_recving.update(failed)
        return output

    def balance_device(self, protected):
        self.scheduler.balance_device(protected)

    def request_finished(self, request, block_ids):
        self.scheduler.finish_request(request.request_id)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    def has_pending_block_frees(self):
        return self.scheduler.awaiting_spill()

    def has_pending_push_work(self):
        return self.scheduler.busy()

    def reset_cache(self):
        return self.scheduler.reset()

    def register_kv_caches(self, kv_caches):
        self.worker.register_kv_caches(kv_caches)

    def get_handshake_metadata(self) -> BtcHostHandshakeMetadata:
        return BtcHostHandshakeMetadata(self.worker.rank, self.worker.host_backing_identity())

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.scheduler.finish_host_attachment(metadata)

    def flush_stores_before_update(self, metadata: BtcTpuMetadata):
        assert isinstance(metadata, BtcTpuMetadata)
        self._transfer_jobs(metadata, load=False)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, BtcTpuMetadata)
        self._transfer_jobs(metadata, load=True)

    def _transfer_jobs(self, metadata: BtcTpuMetadata, load: bool):
        from block_tree_cache import WorkKind

        kind = WorkKind.HOST_TO_DEVICE if load else WorkKind.DEVICE_TO_HOST
        rank = self.worker.rank
        submitted = []
        for operation, work in metadata.jobs.items():
            if work.kind != kind or rank not in work.participants:
                continue
            pages = []
            for item in work.pages:
                device, host = (
                    (item.target, item.source) if load else (item.source, item.target)
                )
                pages.append((item.position.group, device[0].slot, host[0].slot))
            job_id = operation.operation
            self.worker.submit(job_id, load, pages)
            self._rank_ops[job_id] = operation
            submitted.append(job_id)
        # Stores must finish before CoW or zeroing can modify the carriers.
        # Loads run after CoW/zeroing, before forward can read or donate a pool.
        self.worker.wait(submitted)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_worker_meta(self):
        receipts = {}
        for job_id, result in self.worker.get_finished():
            operation = self._rank_ops[job_id]
            receipts[operation] = {self.worker.rank: result}
            if result.safe:
                del self._rank_ops[job_id]
        return BtcRankReceipts(receipts) if receipts else None

    def shutdown(self):
        if self.role is KVConnectorRole.SCHEDULER:
            self.scheduler.close()
        else:
            self.worker.close()
