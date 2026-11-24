"""Planning for fixed-batch decode grids and distribution workloads."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rtp_llm.test.perf_test.config_schema import (
    DecodeDistributionWorkloadConfig,
    DecodeGridWorkloadConfig,
    EngineConfig,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeCaseSpec,
    ExactBatchPolicy,
    FixedBatchConfig,
)
from rtp_llm.test.perf_test.decode.scheduler import (
    schedule_decode_requests,
    scheduler_local_batch_size,
)
from rtp_llm.test.perf_test.errors import PlanningError

DecodeWorkloadConfig = DecodeGridWorkloadConfig | DecodeDistributionWorkloadConfig

# Must match kMaxNumSteps in cpp/engine_base/TorchProfiler.cc. The request's
# profile_step gate and the engine's capture window must have the same width.
_MAX_PROFILE_STEPS = 1000


def validate_decode_profile_steps(
    steps: int, *, output_length: int, engine: EngineConfig
) -> None:
    """Reject windows that cannot safely close within one capture round."""
    if type(steps) is not int or steps <= 0:
        raise PlanningError("workload.profiling.steps must be a positive integer")
    max_steps = min(output_length, _MAX_PROFILE_STEPS)
    if steps > max_steps:
        raise PlanningError(
            "workload.profiling.steps must not exceed "
            f"{max_steps} per round (output length and engine limit {_MAX_PROFILE_STEPS})"
        )
    sp_type = engine.env.get("SP_TYPE", "")
    for name, value in engine.args.items():
        # CLI wins over its paired env, including argparse's accepted prefixes.
        if "sp_type".startswith(name):
            sp_type = value
    if str(sp_type).strip().lower() not in {"", "none"} and steps != 1:
        # SP/MTP advances the profiler once per process(), not per output token.
        # Proposal width alone does not bound every model's token stride/tree.
        raise PlanningError(
            "SP/MTP decode profiling currently requires steps=1; output tokens "
            "do not establish a model-independent iteration budget"
        )


def _validate_capacity(
    inputs: Sequence[int],
    output: int,
    engine: EngineConfig,
    dp_size: int,
) -> None:
    if any(input_length + output > engine.max_seq_len for input_length in inputs):
        raise PlanningError("decode input plus output exceeds engine.max_seq_len")
    schedule = schedule_decode_requests(inputs, dp_size=dp_size)
    scheduler_local_batch_size(
        tuple(worker.local_batch_size for worker in schedule.workers)
    )
    for worker in schedule.workers:
        if worker.local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "decode worker-local batch exceeds engine.concurrency_limit"
            )
        if sum(worker.input_lengths) > engine.max_batch_tokens:
            raise PlanningError(
                "decode worker-local input exceeds engine.max_batch_tokens"
            )


def _spec(
    workload: DecodeWorkloadConfig,
    inputs: Sequence[int],
    output: int,
    policy: ExactBatchPolicy,
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
        batch_unit=workload.batch.unit,
    )


def validate_decode_frontend_lengths(
    cases: Sequence[Mapping[str, Any]], *, dp_size: int, local_dp_size: int
) -> None:
    """Check each executable batch against its frontend-local DP allocations."""
    if local_dp_size <= 1:
        return
    for case in cases:
        spec = DecodeCaseSpec.from_identity(case)
        workers = schedule_decode_requests(spec.input_lengths, dp_size=dp_size).workers
        for offset in range(0, dp_size, local_dp_size):
            lengths = {
                length
                for worker in workers[offset : offset + local_dp_size]
                for length in worker.input_lengths
            }
            if len(lengths) > 1:
                raise PlanningError(
                    "mixed decode input lengths with multiple DP workers on one "
                    "frontend require explicit DP routing; use one DP worker per "
                    "node or uniform input lengths"
                )


def plan_decode_cases(
    workload: DecodeWorkloadConfig,
    engine: EngineConfig,
    *,
    dp_size: int,
    resolved_workload: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Freeze ordered decode cases and reject known engine capacity violations."""
    if workload.profiling.enabled and dp_size != 1:
        raise PlanningError("decode profiling requires DP=1")
    if workload.profiling.enabled:
        min_output = (
            min(workload.output_lengths)
            if isinstance(workload, DecodeGridWorkloadConfig)
            else workload.output_length
        )
        validate_decode_profile_steps(
            workload.profiling.steps, output_length=min_output, engine=engine
        )
    cases = []
    if isinstance(workload, DecodeGridWorkloadConfig):
        for input_length in workload.input_lengths:
            for output_length in workload.output_lengths:
                for size in workload.batch.sizes:
                    global_size = (
                        size * dp_size if workload.batch.unit == "per_dp_rank" else size
                    )
                    inputs = (input_length,) * global_size
                    _validate_capacity(
                        inputs,
                        output_length,
                        engine,
                        dp_size,
                    )
                    cases.append(
                        _spec(
                            workload,
                            inputs,
                            output_length,
                            ExactBatchPolicy(strategy="fixed", size=global_size),
                        )
                    )
    elif isinstance(workload, DecodeDistributionWorkloadConfig):
        if resolved_workload is None or "distribution" not in resolved_workload:
            raise PlanningError("resolved distribution content is required")
        materialized = parse_distribution_inputs(
            resolved_workload["distribution"], workload.batch
        )
        for inputs, source_kind in materialized:
            _validate_capacity(
                inputs,
                workload.output_length,
                engine,
                dp_size,
            )
            cases.append(
                _spec(
                    workload,
                    inputs,
                    workload.output_length,
                    ExactBatchPolicy(strategy="fixed", size=len(inputs)),
                    distribution={"source_kind": source_kind},
                )
            )
    else:
        raise PlanningError("decode planner requires grid or distribution workload")
    if not cases:
        raise PlanningError("decode workload generated no cases")
    return tuple(spec.to_identity() for spec in cases)


# ---------------------------------------------------------------------------
# Materialized decode distribution inputs (merged from decode/distribution.py).
# ---------------------------------------------------------------------------

from rtp_llm.test.perf_test.errors import PlanningError


def _lengths(path: str, value: Any) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PlanningError(f"{path} must be a non-empty array")
    parsed = []
    for index, item in enumerate(value):
        if type(item) is not int or item <= 0:
            raise PlanningError(f"{path}[{index}] must be a positive integer")
        parsed.append(item)
    return tuple(parsed)


def parse_distribution_inputs(
    distribution: Any, batch: FixedBatchConfig
) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    """Materialize input prefixes or exact maps for each fixed batch size."""
    if isinstance(distribution, list):
        sequence_lengths = _lengths("distribution", distribution)
    elif isinstance(distribution, Mapping) and set(distribution) == {
        "sequence_lengths"
    }:
        sequence_lengths = _lengths(
            "distribution.sequence_lengths", distribution["sequence_lengths"]
        )
    else:
        sequence_lengths = ()
    if sequence_lengths:
        required = max(batch.sizes)
        if len(sequence_lengths) < required:
            raise PlanningError("distribution does not contain enough sequence lengths")
        return tuple(
            (sequence_lengths[:size], "sequence_lengths") for size in batch.sizes
        )
    if (
        not isinstance(distribution, Mapping)
        or "batch_seq_len_map" not in distribution
        or set(distribution)
        - {"batch_seq_len_map", "description", "source_csv", "sampling_params"}
    ):
        raise PlanningError(
            "distribution must contain sequence_lengths or batch_seq_len_map"
        )
    raw_map = distribution["batch_seq_len_map"]
    expected_keys = {str(size) for size in batch.sizes}
    if not isinstance(raw_map, Mapping) or not expected_keys.issubset(raw_map):
        raise PlanningError(
            "distribution batch map must contain every fixed batch size"
        )
    result = []
    for size in batch.sizes:
        lengths = _lengths(f"distribution.batch_seq_len_map.{size}", raw_map[str(size)])
        if len(lengths) != size:
            raise PlanningError(
                f"distribution batch {size} does not match its sequence count"
            )
        result.append((lengths, "batch_seq_len_map"))
    return tuple(result)
