"""Batch selection policies independent of decode transport and persistence."""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, Tuple, Union

from rtp_llm.test.perf_test.decode.executor import DecodeCapacityError
from rtp_llm.test.perf_test.decode.metrics import validate_decode_measurement
from rtp_llm.test.perf_test.decode.model import (
    CaseBatchPolicy,
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    DecodeStrategyOutcome,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)

Measure = Callable[[DecodeCaseSpec, int], DecodeBatchMeasurement]


def _candidates(max_size: int, step: int) -> Tuple[int, ...]:
    values = [1]
    values.extend(range(step, max_size + 1, step))
    values.append(max_size)
    return tuple(sorted(set(values)))


def _error_text(error: Exception) -> str:
    return " ".join(str(error).splitlines())[:500]


def _summarize(measurement: DecodeBatchMeasurement, target: float) -> Dict[str, Any]:
    requests = [
        request for run in measurement.measurements for request in run["requests"]
    ]
    successful = [request for request in requests if request["success"]]
    success_rate = len(successful) / len(requests) if requests else 0.0
    tpot = (
        statistics.fmean(request["latency_ms"]["tpot"] for request in successful)
        if successful
        else None
    )
    throughput = statistics.fmean(
        run["throughput_tokens_per_second"] for run in measurement.measurements
    )
    satisfied = success_rate == 1.0 and tpot is not None and tpot <= target
    return {
        "batch_size": measurement.batch_size,
        "outcome": "satisfied" if satisfied else "target_exceeded",
        "tpot_ms": tpot,
        "success_rate": success_rate,
        "throughput_tokens_per_second": throughput,
    }


class FixedBatchStrategy:
    def execute(self, case: DecodeCaseSpec, measure: Measure) -> DecodeStrategyOutcome:
        selected = measure(case, case.batch_policy.size)
        return DecodeStrategyOutcome(selected=selected, search=None)


class TargetTPOTStrategy:
    def execute(self, case: DecodeCaseSpec, measure: Measure) -> DecodeStrategyOutcome:
        policy = case.batch_policy
        candidates = _candidates(policy.max_size, policy.candidate_step)
        cache: Dict[int, DecodeBatchMeasurement] = {}
        observations = []
        best = None
        low, high = 0, len(candidates) - 1
        while low <= high:
            middle = (low + high) // 2
            batch_size = candidates[middle]
            try:
                if batch_size not in cache:
                    cache[batch_size] = measure(case, batch_size)
                validate_decode_measurement(
                    case.to_identity(), cache[batch_size], batch_size=batch_size
                )
                observation = _summarize(cache[batch_size], policy.target_tpot_ms)
            except DecodeCapacityError as error:
                observation = {
                    "batch_size": batch_size,
                    "outcome": "capacity_failed",
                    "error": _error_text(error),
                }
            except Exception as error:
                observation = {
                    "batch_size": batch_size,
                    "outcome": "measurement_failed",
                    "error": _error_text(error),
                }
            observations.append(observation)
            if observation["outcome"] == "satisfied":
                best = cache[batch_size]
                low = middle + 1
            else:
                high = middle - 1
        search = {
            "target_tpot_ms": policy.target_tpot_ms,
            "max_size": policy.max_size,
            "candidate_step": policy.candidate_step,
            "outcome": "found" if best is not None else "no_satisfying_candidate",
            "best_batch_size": None if best is None else best.batch_size,
            "observations": observations,
        }
        return DecodeStrategyOutcome(
            selected=best,
            search=search,
            error_code=None if best is not None else "target_tpot_not_met",
        )


def strategy_for(
    policy: CaseBatchPolicy,
) -> Union[FixedBatchStrategy, TargetTPOTStrategy]:
    if isinstance(policy, ExactBatchPolicy):
        return FixedBatchStrategy()
    if isinstance(policy, TargetTPOTPolicy):
        return TargetTPOTStrategy()
    raise TypeError(f"unsupported decode batch policy: {type(policy).__name__}")
