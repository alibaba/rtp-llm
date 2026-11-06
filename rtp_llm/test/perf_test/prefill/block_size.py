"""Shared cache-probe block expectation for planning and engine sessions."""

from typing import Any, Mapping, Optional

from rtp_llm.test.perf_test.errors import PlanningError


def resolve_cache_block_size(
    probe_block_size: Optional[int],
    cache_alignment: int,
    engine_args: Mapping[str, Any],
    engine_env: Mapping[str, Any],
) -> int:
    # CLI arguments override environment defaults in the engine argument parser.
    sharded = engine_args.get(
        "prefill_cp_kv_cache_sharded",
        engine_env.get("PREFILL_CP_KV_CACHE_SHARDED", "0"),
    )
    if (
        str(sharded).lower() in ("yes", "true", "t", "1", "on")
        and probe_block_size is None
    ):
        # Descriptor-specific cache-key strides need not equal base block * CP.
        raise PlanningError(
            "CP-sharded cache requires explicit workload.probe_block_size"
        )
    value = (
        probe_block_size
        if probe_block_size is not None
        else engine_args.get("seq_size_per_block", cache_alignment)
    )
    if type(value) is not int or value <= 0:
        raise PlanningError(
            "probe_block_size/seq_size_per_block must be a positive integer"
        )
    return value
