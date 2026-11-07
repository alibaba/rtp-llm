from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass, field

import msgspec
from navi_sched.asyncio import AsyncNotification

from .SGLangConfig import SGLangConfig
from .SGLangForwardProtocol import SGLangForwardCodec

logger = logging.getLogger(__name__)


@dataclass
class _ForwardBatch:
    native: object
    handle: object
    stats: dict = field(default_factory=dict)
    remaining: set = field(default_factory=set)
    scheduler_seen: set = field(default_factory=set)
    released: bool = False


class SGLangRuntime:
    def __init__(self, manager, original_generate):
        self.manager = manager
        self.original_generate = original_generate
        self.config = SGLangConfig.from_env()
        self.codec = SGLangForwardCodec(self.config.max_payload_bytes)
        self.agent = None
        self.bridge = None
        self.transport = None
        self._notification = None
        self._execution_tasks = {}
        self._batches = {}
        self._closed = False
        self.ready = asyncio.Event()
        self.snapshot = {
            "scheduler": None,
            "agent_workload": None,
            "agent_workload_accepted": False,
            "forwarded_request_count": 0,
            "prefill_completed_request_count": 0,
            "response_frame_count": 0,
        }

    def receive(self, message):
        values = msgspec.msgpack.decode(message.data)
        kind = values["kind"]
        if kind == "ready":
            if self.agent is not None:
                raise RuntimeError(
                    "multiple SGLang schedulers cannot share one NaviSched Agent"
                )
            from navi_sched import NaviSchedAgent
            from navi_sched.util import naviSchedModule

            self.agent = NaviSchedAgent(
                self.config.target_biz,
                server_port=self.config.agent_port or self.manager.server_args.port,
                server_ip=self.config.server_ip,
                total_blocks=values["total_blocks"],
                block_size=values["page_size"],
                linear_step=values["linear_step"],
                hybrid_tail_required=values["hybrid_tail_required"],
                pd_separated=False,
                latency_prediction_mode="unified",
                latency_model_fingerprint=values["latency_fingerprint"],
                latency_history_fingerprint=values["latency_fingerprint"],
                latency_history_binary_fingerprint="sglang==0.5.23",
                runtime_config=self.config.runtime,
                gig={"grpc_port": self.config.agent_grpc_port},
                wait_no_query=False,
            )
            try:
                self.agent.init()
                self.bridge = naviSchedModule()._VLLMAgentBridge(self.agent)
                if not self.bridge.enable_forwarding():
                    raise RuntimeError("failed to enable SGLang NaviSched forwarding")
            except BaseException:
                self.agent.stop()
                raise
            self.snapshot["cache_policy"] = values
            self._notification = AsyncNotification(
                self.bridge.notification_fd(), self._drain_agent, self._fail_agent
            )
            self.ready.set()
        elif kind == "cache":
            if self.bridge is None:
                raise RuntimeError(
                    "SGLang cache update arrived before Agent initialization"
                )
            self.bridge.report_cache_inserted(values["inserted"])
            self.bridge.report_cache_removed(values["removed"])
        elif kind == "cache_ready":
            self.bridge.report_cache_updated(*values["values"])
        elif kind == "workload":
            self.snapshot["scheduler"] = values["values"]
            self.snapshot["scheduler_request_ids"] = values["request_ids"]
            self._report_workload()
        elif kind == "execution":
            self.bridge.report_execution_step(
                values["elapsed_ms"], values["prefill_tokens"], values["decode_tokens"]
            )
            self.snapshot["execution"] = values
        elif kind == "execution_started":
            self.snapshot["execution_started"] = {
                **values,
                "accepted": bool(
                    self.bridge.begin_backend_execution(values["request_count"])
                ),
            }
        elif kind == "execution_finished":
            self.snapshot["execution_finished_accepted"] = bool(
                self.bridge.end_backend_execution()
            )
        elif kind == "stopped":
            self.close()
        else:
            raise ValueError("unknown SGLang observer message")

    def _report_workload(self, admitted=None):
        if self.bridge is None or self.snapshot["scheduler"] is None:
            return
        values = dict(self.snapshot["scheduler"])
        request_ids = set(self.snapshot.get("scheduler_request_ids", ()))
        for state in self._batches.values():
            state.scheduler_seen.update(
                request_id
                for request_id in state.native.request_ids
                if "navi_sched_" + str(request_id) in request_ids
            )
            state.scheduler_seen.update(state.stats)
            if state.scheduler_seen != set(state.native.request_ids):
                # The core keeps the complete native batch in its workload until
                # all of that batch is represented by the scheduler snapshot.
                self.snapshot["agent_workload_deferred_reason"] = (
                    "native batch is still being prepared"
                )
                return
        self.snapshot["agent_workload_deferred_reason"] = ""
        values["covered_native_batch_ids"] = [int(key) for key in self._batches]
        if admitted is not None:
            native = admitted.native
            values.update(
                admitted_schedule_batch_id=int(native.schedule_batch_id),
                admitted_local_batch_id=int(native.batch_id),
                admitted_request_count=len(native.request_ids),
                admitted_pending_cache_tracked=bool(native.pending_cache_tracked),
                admitted_request_ids=list(native.request_ids),
            )
        self.snapshot["agent_workload"] = values
        self.snapshot["agent_workload_accepted"] = bool(
            self.bridge.report_workload(**values)
        )

    def _drain_agent(self):
        loop = asyncio.get_running_loop()
        for _ in range(32):
            for request_id in self.bridge.take_cancelled_requests():
                task = self._execution_tasks.get(int(request_id))
                if task is not None:
                    task.cancel()
            batch = self.bridge.take_next_forward_batch(0)
            if batch is None:
                return False
            if (
                len(self._execution_tasks) + len(batch.request_ids)
                > self.config.max_inflight_requests
            ):
                for request_id in batch.request_ids:
                    self.bridge.fail_forward_response(
                        request_id, "SGLang target is at capacity"
                    )
                self.bridge.release_forward_batch(batch)
                continue
            handle = self.bridge.start_forward_batch(batch, list(batch.request_ids))
            if handle is None:
                self.bridge.release_forward_batch(batch)
                continue
            state = _ForwardBatch(batch, handle, remaining=set(batch.request_ids))
            self._batches[int(batch.batch_id)] = state
            for request_id, payload in zip(batch.request_ids, batch.payloads):
                self.snapshot["forwarded_request_count"] += 1
                task = loop.create_task(self._execute(request_id, payload, state))
                self._execution_tasks[int(request_id)] = task
        return True

    def _fail_agent(self, error):
        logger.exception("SGLang NaviSched Agent stopped forwarding")
        self.snapshot["forwarding_error"] = str(error)
        self.close()

    async def _execute(self, request_id, payload, state):
        obj = None
        completed = False
        try:
            obj = self.codec.decode_request(payload)
            source_rid = obj.rid
            obj.rid = "navi_sched_" + str(request_id)
            obj.stream = True
            # A single destination executes every item of a NaviSched batch. The
            # shared original method bypasses source routing but retains SGLang's
            # native validation, media processing, LoRA loading and detokenization.
            async with asyncio.timeout(self.config.request_timeout_ms / 1000):
                async for response in self.original_generate(self.manager, obj, None):
                    meta = response.get("meta_info", {})
                    meta["id"] = source_rid
                    if request_id not in state.stats:
                        state.stats[request_id] = (
                            int(request_id),
                            int(meta["prompt_tokens"]),
                            int(meta.get("cached_tokens", 0)),
                        )
                        if len(state.stats) == len(state.native.request_ids):
                            self._finish_prefill(state)
                    completed = meta.get("finish_reason") is not None
                    if not self.bridge.publish_forward_response(
                        request_id, [self.codec.encode_response(response)], completed
                    ):
                        raise asyncio.CancelledError()
                    self.snapshot["response_frame_count"] += 1
            if not completed:
                raise RuntimeError("SGLang stream ended without a finish reason")
        except asyncio.CancelledError:
            self.bridge.fail_forward_response(request_id, "SGLang request cancelled")
            raise
        except Exception as error:
            self.bridge.fail_forward_response(request_id, str(error))
        finally:
            if obj is not None:
                self.manager.abort_request(obj.rid)
            self._execution_tasks.pop(int(request_id), None)
            state.remaining.discard(request_id)
            if not completed and state.handle is not None:
                if len(state.native.request_ids) == 1:
                    self._release_batch(state)
                elif self.bridge.abort_forward_batch_requests(
                    state.native, [request_id]
                ):
                    state.stats.pop(request_id, None)
                    state.scheduler_seen.discard(request_id)
                    if len(state.stats) == len(state.native.request_ids):
                        self._finish_prefill(state)
                else:
                    self._release_batch(state)
                    for remaining_id in state.remaining:
                        task = self._execution_tasks.get(int(remaining_id))
                        if task is not None:
                            task.cancel()
            if not state.remaining:
                self._release_batch(state)
                self._report_workload()

    def _finish_prefill(self, state):
        # NaviSched tracks prompt work through the first generated token.
        # Decode remains in the scheduler's observed workload.
        self._report_workload(admitted=state)
        if not self.bridge.finish_batch(state.handle, list(state.stats.values())):
            raise RuntimeError("NaviSched rejected SGLang prefill completion")
        self.snapshot["prefill_completed_request_count"] += len(state.stats)
        self._release_batch(state)

    def _release_batch(self, state):
        if state.released:
            return
        self.bridge.release_forward_batch(state.native)
        state.handle = None
        state.released = True
        self._batches.pop(int(state.native.batch_id), None)

    async def generate(self, obj, request):
        from .SGLangTransport import SGLangTransport

        if self._closed:
            raise RuntimeError("SGLang NaviSched runtime is closed")
        self.manager.auto_create_handle_loop()
        await asyncio.wait_for(self.ready.wait(), self.config.request_timeout_ms / 1000)
        if self.transport is None:
            self.transport = SGLangTransport(
                self.manager,
                self.config,
                page_size=self.snapshot["cache_policy"]["page_size"],
            )
            self.snapshot["transport"] = self.transport.snapshot
        async with aclosing(
            self.transport.generate(obj, request, self.original_generate)
        ) as responses:
            async for output in responses:
                yield output

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.transport is not None:
            self.transport.close()
        if self._notification is not None:
            self._notification.close()
        for request_id, task in tuple(self._execution_tasks.items()):
            task.cancel()
            self.manager.abort_request("navi_sched_" + str(request_id))
        for state in tuple(self._batches.values()):
            self._release_batch(state)
        if self.bridge is not None:
            self.bridge.disable_forwarding()
        if self.agent is not None:
            self.agent.stop()
