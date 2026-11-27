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

"""Adapt the vLLM scheduler that owns a BTC instance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from btc_vllm.scheduler import step_device_slots

if TYPE_CHECKING:
    from btc_vllm.connector import BtcTpuConnector
    from btc_vllm.multi_connector import BtcProducerMultiConnector
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request


class BtcSchedulerAdapter:
    def __init__(
        self,
        scheduler: Scheduler,
        connector: BtcTpuConnector | BtcProducerMultiConnector,
    ) -> None:
        self._scheduler = scheduler
        self._connector = connector
        self._committing: dict[str, Request] | None = None
        self._original_update = scheduler.update_from_output
        self._original_handle_stopped = scheduler._handle_stopped_request
        self._original_handle_invalid = scheduler._handle_invalid_blocks
        self._original_finish_load = scheduler._update_waiting_for_remote_kv
        scheduler._get_local_prefix_cache_hit = self._get_local_prefix_cache_hit
        scheduler.update_from_output = self._update_from_output
        scheduler._handle_stopped_request = self._handle_stopped_request
        scheduler._handle_invalid_blocks = self._handle_invalid_blocks
        scheduler._update_waiting_for_remote_kv = self._update_waiting_for_remote_kv

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        blocks, device_tokens = self._connector.try_joint_prefix_match(
            request, max_cache_hit_length
        )
        return blocks, device_tokens, 0, False

    def _publish(self, request: Request) -> None:
        self._connector.publish_computed_gpu_blocks(
            request, request.num_computed_tokens - request.num_in_flight_tokens
        )

    def _update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        protected = step_device_slots(scheduler_output)
        connector_output = self._connector.prepare_scheduler_output(
            model_runner_output.kv_connector_output, protected
        )
        model_runner_output.kv_connector_output = connector_output
        failed = () if connector_output is None else connector_output.failed_recving
        requests = self._scheduler.requests
        # vLLM consumes the stale-token count while processing this output.
        # Capture eligibility before it can reach zero on the last stale step.
        committing = {
            req_id: request
            for req_id in scheduler_output.num_scheduled_tokens
            if (request := requests.get(req_id)) is not None
            and not request.is_finished()
            and not request.num_stale_output_tokens
            and req_id not in failed
        }
        assert self._committing is None, "BTC scheduler output processing is reentrant"
        self._committing = committing
        try:
            outputs = self._original_update(scheduler_output, model_runner_output)
            for req_id, request in committing.items():
                if requests.get(req_id) is request and not request.is_finished():
                    self._publish(request)
            self._connector.balance_device(protected)
            return outputs
        finally:
            self._committing = None

    def _handle_stopped_request(self, request: Request) -> bool:
        # A terminal or streaming stop can release or revise the request's pages.
        committing = self._committing
        if (
            committing is not None
            and committing.pop(request.request_id, None) is request
        ):
            self._publish(request)
        return self._original_handle_stopped(request)

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        skipped_ids = self._original_handle_invalid(
            invalid_block_ids, num_scheduled_tokens
        )
        committing = self._committing
        if committing is not None:
            for req_id in skipped_ids:
                committing.pop(req_id, None)
        return skipped_ids

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        failed = request.request_id in self._scheduler.failed_recving_kv_req_ids
        committed = request.num_computed_tokens
        self._original_finish_load(request)
        if not failed:
            self._connector.publish_computed_gpu_blocks(request, committed)
