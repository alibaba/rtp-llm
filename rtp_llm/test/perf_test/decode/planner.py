"""Unified planning for exact-batch and target-TPOT decode workloads."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rtp_llm.test.perf_test.config_schema import (
    DecodeDistributionWorkloadConfig,
    DecodeGridWorkloadConfig,
    EngineConfig,
)
from rtp_llm.test.perf_test.decode.distribution import parse_distribution_inputs
from rtp_llm.test.perf_test.decode.model import (
    CaseBatchPolicy,
    DecodeBatchConfig,
    DecodeCaseSpec,
    ExactBatchPolicy,
    FixedBatchConfig,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.errors import PlanningError

DecodeWorkloadConfig = DecodeGridWorkloadConfig | DecodeDistributionWorkloadConfig


def _validate_capacity(
    inputs: Sequence[int], output: int, engine: EngineConfig, dp_size: int
) -> None:
    if any(input_length + output > engine.max_seq_len for input_length in inputs):
        raise PlanningError("decode input plus output exceeds engine.max_seq_len")
    schedule = schedule_decode_requests(inputs, dp_size=dp_size)
    for worker in schedule.workers:
        if worker.local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "decode worker-local batch exceeds engine.concurrency_limit"
            )
        if sum(worker.input_lengths) > engine.max_batch_tokens:
            raise PlanningError(
                "decode worker-local input exceeds engine.max_batch_tokens"
            )


def _policy(batch: DecodeBatchConfig, size: Optional[int] = None) -> CaseBatchPolicy:
    if isinstance(batch, FixedBatchConfig):
        assert size is not None
        return ExactBatchPolicy(strategy="fixed", size=size)
    return TargetTPOTPolicy(
        strategy="target_tpot",
        max_size=batch.max_size,
        candidate_step=batch.candidate_step,
        target_tpot_ms=batch.target_tpot_ms,
    )


def _spec(
    workload: DecodeWorkloadConfig,
    inputs: Sequence[int],
    output: int,
    policy: CaseBatchPolicy,
    distribution: Optional[Dict[str, Any]] = None,
) -> DecodeCaseSpec:
    return DecodeCaseSpec(
        workload_kind=workload.kind,
        input_lengths=tuple(inputs),
        output_length=output,
        batch_policy=policy,
        measure_runs=workload.measure_runs,
        warmup_runs=workload.warmup_runs,
        generate_config=copy.deepcopy(workload.generate_config),
        profiling=workload.profiling,
        distribution=distribution,
    )


def plan_decode_cases(
    workload: DecodeWorkloadConfig,
    engine: EngineConfig,
    *,
    dp_size: int,
    resolved_workload: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Freeze ordered decode cases and reject known engine capacity violations."""
    cases = []
    if isinstance(workload, DecodeGridWorkloadConfig):
        for input_length in workload.input_lengths:
            for output_length in workload.output_lengths:
                sizes = (
                    workload.batch.sizes
                    if isinstance(workload.batch, FixedBatchConfig)
                    else (workload.batch.max_size,)
                )
                for size in sizes:
                    inputs = (input_length,) * size
                    _validate_capacity(inputs, output_length, engine, dp_size)
                    cases.append(
                        _spec(
                            workload,
                            inputs,
                            output_length,
                            _policy(workload.batch, size),
                        )
                    )
    elif isinstance(workload, DecodeDistributionWorkloadConfig):
        if resolved_workload is None or "distribution" not in resolved_workload:
            raise PlanningError("resolved distribution content is required")
        materialized = parse_distribution_inputs(
            resolved_workload["distribution"], workload.batch
        )
        for inputs, source_kind in materialized:
            _validate_capacity(inputs, workload.output_length, engine, dp_size)
            size = len(inputs) if isinstance(workload.batch, FixedBatchConfig) else None
            cases.append(
                _spec(
                    workload,
                    inputs,
                    workload.output_length,
                    _policy(workload.batch, size),
                    distribution={"source_kind": source_kind},
                )
            )
    else:
        raise PlanningError("decode planner requires grid or distribution workload")
    if not cases:
        raise PlanningError("decode workload generated no cases")
    return tuple(spec.to_identity() for spec in cases)
