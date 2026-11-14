"""Transport-independent execution and measurement of a candidate decode batch."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement, DecodeCaseSpec
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler, sanitize_error
from rtp_llm.test.perf_test.decode.scheduler import (
    DecodeBatchSchedule,
    schedule_decode_requests,
    scheduler_local_batch_size,
)
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import PlanningError


class DecodeCapacityError(PlanningError):
    """A decode candidate exceeds available inputs or per-worker engine bounds."""


class DecodeBatchExecutor:
    def __init__(
        self,
        transport: DecodeHttpTransport,
        *,
        engine: EngineConfig,
        dp_size: int,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._engine = engine
        self._dp_size = dp_size
        self._monotonic = monotonic
        self._profile_start_attempted = False

    @property
    def profile_start_attempted(self) -> bool:
        return self._profile_start_attempted

    def _prepare(
        self, case: DecodeCaseSpec, batch_size: int
    ) -> Tuple[Tuple[int, ...], DecodeBatchSchedule]:
        inputs = case.input_lengths[:batch_size]
        if len(inputs) != batch_size:
            raise DecodeCapacityError(
                f"decode batch {batch_size} exceeds available inputs "
                f"({len(case.input_lengths)})"
            )
        schedule = schedule_decode_requests(inputs, dp_size=self._dp_size)
        try:
            scheduler_local_batch_size(
                tuple(worker.local_batch_size for worker in schedule.workers)
            )
        except PlanningError as error:
            raise DecodeCapacityError(str(error)) from error
        for worker in schedule.workers:
            if worker.local_batch_size > self._engine.concurrency_limit:
                raise DecodeCapacityError(
                    f"decode worker {worker.worker_index} batch "
                    f"{worker.local_batch_size} exceeds concurrency_limit "
                    f"{self._engine.concurrency_limit}"
                )
            if sum(worker.input_lengths) > self._engine.max_batch_tokens:
                raise DecodeCapacityError(
                    f"decode worker {worker.worker_index} input tokens "
                    f"{sum(worker.input_lengths)} exceeds max_batch_tokens "
                    f"{self._engine.max_batch_tokens}"
                )
        return inputs, schedule

    def measure(self, case: DecodeCaseSpec, batch_size: int) -> DecodeBatchMeasurement:
        inputs, schedule = self._prepare(case, batch_size)
        self._transport.configure_scheduler(
            tuple(worker.to_dict() for worker in schedule.workers)
        )
        for run_index in range(case.warmup_runs):
            requests = self._transport.run_round(
                input_lengths=inputs,
                output_length=case.output_length,
                generate_config=case.generate_config,
            )
            for request in requests:
                if not request["success"]:
                    raise RuntimeError(
                        f"decode warmup round {run_index} request "
                        f"{request['request_index']} failed: "
                        f"{request.get('error', 'request failed')}"
                    )
        measurements = []
        for run_index in range(case.measure_runs):
            started = self._monotonic()
            requests = self._transport.run_round(
                input_lengths=inputs,
                output_length=case.output_length,
                generate_config=case.generate_config,
            )
            elapsed = self._monotonic() - started
            successful_tokens = sum(
                item["output_length"] for item in requests if item["success"]
            )
            measurements.append(
                {
                    "run": run_index,
                    "throughput_tokens_per_second": (
                        successful_tokens / elapsed if elapsed > 0 else 0.0
                    ),
                    "worker_allocations": [
                        worker.to_dict() for worker in schedule.workers
                    ],
                    "requests": copy.deepcopy(list(requests)),
                }
            )
        return DecodeBatchMeasurement(
            batch_size=batch_size,
            input_lengths=inputs,
            worker_allocations=tuple(worker.to_dict() for worker in schedule.workers),
            measurements=tuple(measurements),
        )

    def profile(
        self,
        case: DecodeCaseSpec,
        batch_size: int,
        *,
        trace_name: str,
        steps: int,
        enable_all_ranks: bool,
    ) -> None:
        self._profile_start_attempted = False
        inputs, schedule = self._prepare(case, batch_size)
        self._transport.configure_scheduler(
            tuple(worker.to_dict() for worker in schedule.workers)
        )
        self._profile_start_attempted = True
        self._transport.start_profile(
            trace_name=trace_name, steps=steps, enable_all_ranks=enable_all_ranks
        )
        profile_config = copy.deepcopy(case.generate_config)
        profile_config.update(
            {
                "gen_timeline": True,
                "profile_step": steps,
                "profile_trace_name": trace_name,
            }
        )
        for run_index in range(case.profiling.runs):
            requests = self._transport.run_round(
                input_lengths=inputs,
                output_length=case.output_length,
                generate_config=profile_config,
            )
            for request in requests:
                if not request["success"]:
                    error = RuntimeError(
                        f"decode profile round {run_index} request "
                        f"{request['request_index']} failed: "
                        f"{request.get('error', 'request failed')}"
                    )
                    raise RuntimeError(sanitize_error(error))


class DecodeCaseExecutor:
    def __init__(
        self,
        batch_executor: DecodeBatchExecutor,
        *,
        topology: Mapping[str, int],
        profiler: Optional[DecodeProfiler] = None,
    ):
        self._batch_executor = batch_executor
        self._topology = dict(topology)
        self._profiler = profiler

    @property
    def session_reusable(self) -> bool:
        return self._profiler is None or self._profiler.session_reusable

    def execute(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.session_reusable:
            raise RuntimeError(
                "decode profiling session requires engine shutdown after incomplete capture"
            )
        # Strategies depend on DecodeCapacityError; resolve after module loading.
        from rtp_llm.test.perf_test.decode.strategy import strategy_for

        identity = {
            name: copy.deepcopy(value)
            for name, value in case.items()
            if name != "case_id"
        }
        spec = DecodeCaseSpec.from_identity(identity)
        outcome = strategy_for(spec.batch_policy, dp_size=self._topology["dp"]).execute(
            spec, self._batch_executor.measure
        )
        if outcome.selected is None:
            return {
                "case_id": case["case_id"],
                "identity": identity,
                "status": "target_tpot_not_met",
                "runs": spec.measure_runs,
                "errors": [
                    {
                        "code": outcome.error_code or "target_tpot_not_met",
                        "message": "no candidate batch satisfied target_tpot_ms",
                        "details": {"search": copy.deepcopy(outcome.search)},
                    }
                ],
                "metrics": None,
            }
        profiling = {"status": "disabled", "manifest_path": None, "error": None}
        result = build_decode_case_result(
            case,
            selected=outcome.selected,
            topology=self._topology,
            search=outcome.search,
            profiling=profiling,
        )
        if result["status"] == "ok" and spec.profiling.enabled:
            if self._profiler is None:
                raise RuntimeError(
                    "decode profiling is enabled but no profiler is configured"
                )
            result["metrics"]["profiling"] = copy.deepcopy(
                self._profiler.profile(spec, outcome.selected.batch_size)
            )
            if not self.session_reusable:
                # Retain formal measurements, but prevent an incomplete capture
                # from producing a completed run eligible as a baseline.
                result["status"] = "request_failed"
                result["errors"].append(
                    {
                        "code": "request_failed",
                        "message": (
                            "decode profile capture did not complete; the engine "
                            "session requires shutdown before another case; "
                            f"{result['metrics']['profiling']['error']}"
                        ),
                    }
                )
        return result
