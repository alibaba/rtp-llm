"""Shared workload case planning, independent of execution backend policy."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from rtp_llm.test.perf_test.config_resolver import ResolvedConfig
from rtp_llm.test.perf_test.config_schema import (
    DecodeDistributionWorkloadConfig,
    DecodeGridWorkloadConfig,
    ExecutionParser,
    PrefillWorkloadConfig,
    parse_config,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases


def plan_cases(
    mode: str,
    source: Mapping[str, Any],
    resolved: ResolvedConfig,
    *,
    execution_parser: Optional[ExecutionParser] = None,
) -> Tuple[Dict[str, object], ...]:
    if mode == "prefill":
        from rtp_llm.test.perf_test.prefill.traffic_batch import (
            materialize_traffic_source,
        )

        source = materialize_traffic_source(
            source, resolved.resolved_config["workload"]
        )
    config = parse_config(source, execution_parser=execution_parser)
    workload = config.workload
    if mode == "prefill":
        if not isinstance(workload, PrefillWorkloadConfig):
            raise ConfigurationError("--mode=prefill requires workload.kind=cache_grid")
        return plan_prefill_cases(
            workload, config.engine, dp_size=config.model.topology.dp
        )
    if isinstance(
        workload, (DecodeGridWorkloadConfig, DecodeDistributionWorkloadConfig)
    ):
        return plan_decode_cases(
            workload,
            config.engine,
            dp_size=config.model.topology.dp,
            resolved_workload=resolved.resolved_config["workload"],
        )
    raise ConfigurationError(
        "--mode=decode requires workload.kind=grid or distribution"
    )
