from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from contextlib import suppress
from typing import Any

logger = logging.getLogger("vllm.v1.engine.core")


class VLLMCoreCallbacks:
    def get_hash_block_size(self, core):
        block_size = self._cache_policy.block_size
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("invalid NaviSched runtime hash block size")
        return block_size

    def get_routing_metadata(self, core):
        from .VLLMCachePolicy import get_vllm_prefill_routing_policy

        policy = get_vllm_prefill_routing_policy(core.scheduler.get_kv_connector())
        return {
            "block_size": self.get_hash_block_size(core),
            "trim_last_prompt_token": policy["trim_last_prompt_token"],
            "prefix_cache_lookup_enabled": bool(
                core.vllm_config.cache_config.enable_prefix_caching
            ),
        }

    def log_core_stage(
        self, core, stage: str, started_ns: int, finished_ns: int, **details: Any
    ) -> None:
        observer = self
        if stage in ("engine_step", "future_result", "batch_queue_future_result"):
            report_stage = getattr(observer, "_report_core_step_stage", None)
            if report_stage is not None:
                with suppress(Exception):
                    report_stage(stage, (finished_ns - started_ns) / 1_000_000)
        if finished_ns - started_ns < 1_000_000_000:
            return
        log_stage = getattr(observer, "_log_slow_engine_stage", None)
        if log_stage is not None:
            try:
                details["has_work"] = core.has_work()
                log_stage("core_" + stage, started_ns, finished_ns, **details)
            except Exception:
                # Diagnostics must not interrupt the EngineCore loop.
                pass

    def sample_tokens(
        self, executor, execution_future, grammar_output, non_block=False
    ):
        output = None
        try:
            output = executor.sample_tokens(grammar_output, non_block=non_block)
            return output
        except BaseException as error:
            output = Future()
            output.set_exception(error)
            raise
        finally:
            self.observe_output_safely(execution_future, output)

    def observe_output_safely(self, execution_future, output):
        try:
            self.observe_model_output(execution_future, output)
        except Exception:
            logger.exception("Failed to record NaviSched model output timing")

    def wait_for_output(self, core, future, scheduler_output, batch_queue=None):
        started_ns = time.perf_counter_ns()
        try:
            return future.result()
        finally:
            details = dict(
                scheduled_requests=len(scheduler_output.num_scheduled_tokens),
                scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            )
            stage = "future_result"
            if batch_queue is not None:
                stage = "batch_queue_future_result"
                details["batch_queue_size"] = len(batch_queue)
            self.log_core_stage(
                core, stage, started_ns, time.perf_counter_ns(), **details
            )

    def process_input_queue(self, core):
        had_work = core.has_work()
        started_ns = time.perf_counter_ns()
        try:
            return core._process_input_queue()
        finally:
            if had_work:
                self.log_core_stage(
                    core, "input_queue", started_ns, time.perf_counter_ns()
                )

    def process_engine_step(self, core):
        started_ns = time.perf_counter_ns()
        try:
            return core._process_engine_step()
        finally:
            self.log_core_stage(core, "engine_step", started_ns, time.perf_counter_ns())
