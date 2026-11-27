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

"""A Raiden prefill producer with BTC as its only prefix-cache owner."""

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorMetadata,
    MultiKVConnectorWorkerMetadata,
)

from btc_vllm.connector import BtcTpuConnector
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector

_COMPOSITIONS = {
    "BtcTpuConnector": "TPURaidenConnector",
    "Qwen4BtcTpuConnector": "Qwen4ExpRaidenConnector",
}


def validate_btc_producer(transfer):
    children = transfer.kv_connector_extra_config["connectors"]
    if (
        transfer.kv_role != "kv_both"
        or len(children) != 2
        or children[0].get("kv_role") != "kv_producer"
        or children[1].get("kv_role") != "kv_both"
        or children[1].get("kv_connector") not in _COMPOSITIONS
        or _COMPOSITIONS[children[1]["kv_connector"]] != children[0].get("kv_connector")
    ):
        raise ValueError(
            "BTC TPU Prefill requires TPUMultiConnector (kv_both) with ordered "
            "Raiden producer and compatible BTC cache child"
        )
    if transfer.kv_load_failure_policy != "recompute":
        raise ValueError(
            "BTC TPU prefill requires kv_load_failure_policy=recompute on the parent"
        )
    if any("engine_id" in child for child in children):
        raise ValueError(
            "Set the BTC prefill engine_id on the parent TPUMultiConnector"
        )
    return children[0]


class BtcProducerMultiConnector(TPUMultiConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        validate_btc_producer(vllm_config.kv_transfer_config)
        super().__init__(vllm_config, role, kv_cache_config)
        self.cache = self._connectors[1]
        assert isinstance(self.cache, BtcTpuConnector)

    def try_joint_prefix_match(self, request, max_cache_hit_length):
        return self.cache.try_joint_prefix_match(request, max_cache_hit_length)

    def publish_computed_gpu_blocks(self, request, num_committed_tokens):
        self.cache.publish_computed_gpu_blocks(request, num_committed_tokens)

    def prepare_scheduler_output(self, output, protected):
        btc_meta = None
        if output is not None and output.kv_connector_worker_meta is not None:
            meta = output.kv_connector_worker_meta
            assert isinstance(meta, MultiKVConnectorWorkerMetadata)
            btc_meta = meta.metadata[1]
        return self.cache.apply_rank_receipts(output, btc_meta, protected)

    def balance_device(self, protected):
        self.cache.balance_device(protected)

    def register_runner(self, runner):
        super().register_runner(runner)
        runner._btc_flush_stores = self.flush_stores_before_update

    def flush_stores_before_update(self, metadata):
        assert isinstance(metadata, MultiKVConnectorMetadata)
        self.cache.flush_stores_before_update(metadata.metadata[1])
