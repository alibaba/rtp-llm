"""Unified prefill planning and session access for an externally owned service."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from rtp_llm.test.perf_test.capabilities import validate_capability
from rtp_llm.test.perf_test.config_resolver import canonical_json, load_resolved_config
from rtp_llm.test.perf_test.config_schema import (
    DEFAULT_PREFILL_TRANSPORT,
    PrefillWorkloadConfig,
    parse_config,
)
from rtp_llm.test.perf_test.entry import _source_config
from rtp_llm.test.perf_test.errors import ConfigurationError, PlanningError
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.probe import (
    probe_cache_block,
    resolve_probe_block_size,
)
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.prefill.traffic_batch import materialize_traffic_source
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def build_external_prefill_plan(
    config_path,
    result_directory,
    actual_topology,
    *,
    resolved_workload_path=None,
    execution_parser=None,
    resource_planner=None,
    capability_validator=validate_capability,
):
    """Build the local prefill case identities without creating run artifacts.

    Mirrors ``build_external_decode_plan``: the configured topology must
    match the topology the running service was started with, and the
    returned plan is fully frozen (with standard case IDs and fingerprints)
    before any run directory is initialized.
    """
    source = _source_config(str(config_path))
    resolved = load_resolved_config(config_path, execution_parser=execution_parser)
    if resolved_workload_path:
        delivered = json.loads(Path(resolved_workload_path).read_text(encoding="utf-8"))

        def without_sampling(workload):
            workload = copy.deepcopy(workload)
            workload.pop("traffic_sampling", None)
            for grid in workload.get("grids", ()):
                grid["workload"] = without_sampling(grid["workload"])
            return workload

        if canonical_json(without_sampling(delivered)) != canonical_json(
            without_sampling(resolved.resolved_config["workload"])
        ):
            raise PlanningError(
                "delivered resolved workload does not match frozen config"
            )
        normalized = {**resolved.resolved_config, "workload": delivered}
        resolved = replace(
            resolved,
            resolved_config=normalized,
            workload_fingerprint=hashlib.sha256(
                canonical_json(delivered).encode("utf-8")
            ).hexdigest(),
        )
    planned_source = materialize_traffic_source(
        source, resolved.resolved_config["workload"]
    )
    config = parse_config(planned_source, execution_parser=execution_parser)
    if not isinstance(config.workload, PrefillWorkloadConfig):
        raise ConfigurationError("external prefill requires workload.kind=cache_grid")
    configured = config.model.topology.as_tuple()
    actual = tuple(actual_topology.get(name) for name in ("tp", "dp", "pp", "ep"))
    if any(type(value) is not int or value <= 0 for value in actual):
        raise PlanningError(
            "runtime topology must supply positive integer TP, DP, PP, and EP sizes"
        )
    if configured != actual:
        raise PlanningError(
            f"runtime topology {actual} does not match configured topology {configured}"
        )
    cases = plan_prefill_cases(
        config.workload, config.engine, dp_size=config.model.topology.dp
    )
    plan = build_execution_plan(
        resource_planner=resource_planner,
        capability_validator=capability_validator,
        mode="prefill",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(result_directory),
    )
    return plan, source


class ExternalPrefillSession:
    """Prefill-runner session handle for a service another lifecycle owns.

    Satisfies the session contract the prefill runner consumes (``client``,
    ``diagnostics``, context-manager ``close``) without ever starting or
    stopping a process: the gang worker owns the model service.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topology: Dict[str, int],
        transport: str = DEFAULT_PREFILL_TRANSPORT,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.topology = dict(topology)
        if transport not in ("http_text", "dashsc_input_ids"):
            raise ValueError(f"unsupported prefill transport: {transport}")
        self._client = HttpPrefillClient(self.endpoint)
        if transport == "dashsc_input_ids":
            from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                DashScPrefillClient,
            )

            self._client = DashScPrefillClient(self._client)
            try:
                self._client.wait_ready(600)
            except BaseException:
                self._client.close()
                raise
        self._diagnostics = EngineDiagnostics()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def diagnostics(self) -> EngineDiagnostics:
        return self._diagnostics

    def close(self) -> None:
        """Release the inference channel; the caller owns the service process."""
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "ExternalPrefillSession":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> Optional[bool]:
        self.close()
        return None


def execute_master_prefill(
    plan,
    *,
    source_config,
    endpoint,
    tokenizer_path,
    server_log_path,
    profiler_directory,
):
    """Run the shared prefill measurement against a caller-owned rank group."""
    del server_log_path, profiler_directory
    root = Path(plan.result_directory)
    run_id = os.environ.get("PERF_TEST_RUN_ID", root.name)
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    workload = plan.resolved_config["workload"]
    model = plan.resolved_config["model"]
    session = None
    try:
        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        tokenizer = _load_tokenizer(
            tokenizer_path,
            checkpoint_path=str(model["checkpoint"]),
            model_type=str(model["type"]),
        )
        commit_tail = int(workload["commit_tail_tokens"])
        block_size = resolve_probe_block_size(source_config)
        session = ExternalPrefillSession(
            endpoint,
            topology=dict(model["topology"]),
            transport=workload.get("transport", DEFAULT_PREFILL_TRANSPORT),
        )
        adapter = PrefillAdapter(
            lambda context: PrefillRunner(
                context.session.client,
                tokenizer,
                commit_tail_tokens=commit_tail,
                cache_block_size=block_size,
            ),
            probe=lambda context: probe_cache_block(
                context.session.client, tokenizer, block_size, commit_tail
            ),
        )
        result = adapter(ModeExecutionContext(plan=plan, session=session, store=store))
        finalize_prefill_run(
            root,
            plan=plan,
            run_id=run_id,
            case_results=result.case_results,
            terminal_state=result.state,
        )
        generate_prefill_report(root, acquire_lock=False)
    except Exception:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    finally:
        if session is not None:
            session.close()
    manifest["state"] = result.state.value
    store.write_manifest(manifest)
    return result
