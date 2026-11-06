"""Load and execute one AOne prefill shard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineDiagnostics,
    EngineInterrupted,
)
from rtp_llm.test.perf_test.modes import ModeExecutionContext
from rtp_llm.test.perf_test.plan import (
    PLAN_SCHEMA_VERSION,
    AOneResourcePlan,
    ExecutionPlan,
    PlannedCase,
    case_id,
    freeze,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json


@dataclass(frozen=True)
class RemoteInputs:
    run_id: str
    shard_id: str
    config_fingerprint: str
    workload_fingerprint: str
    config: Dict[str, Any]
    owned_case_ids: Tuple[str, ...]
    owned_cases: Tuple[Dict[str, Any], ...]


def resolve_remote_runtime_inputs(
    inputs: RemoteInputs,
    *,
    fetcher: Optional[Callable[[str], str]] = None,
    hub_fetcher: Optional[Callable[[str], str]] = None,
) -> RemoteInputs:
    """Resolve external model references without changing collected identity."""
    if fetcher is None:
        from rtp_llm.utils.fuser import fetch_remote_file_to_local

        fetcher = fetch_remote_file_to_local
    config = copy.deepcopy(inputs.config)
    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigurationError("remote resolved config model must be an object")
    resolved: Dict[str, str] = {}
    for key in ("checkpoint", "tokenizer"):
        value = model.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"remote model.{key} must be a non-empty string")
        if value not in resolved:
            expanded = os.path.expanduser(value.strip())
            candidate = fetcher(expanded)
            if candidate == expanded and not os.path.exists(candidate):
                from rtp_llm.test.perf_test.hub_download import is_model_repo_id

                if is_model_repo_id(expanded):
                    if hub_fetcher is None:
                        from huggingface_hub import snapshot_download

                        hub_fetcher = snapshot_download
                    candidate = hub_fetcher(expanded)
            resolved[value] = candidate
        model[key] = resolved[value]
    return replace(inputs, config=config)


def _load_object(path: Union[str, Path], name: str) -> Dict[str, Any]:
    artifact = Path(path)
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"cannot read {name} runfile {artifact}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} runfile must contain an object")
    return value


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_remote_inputs(
    *,
    resolved_config: Union[str, Path],
    resolved_workload: Union[str, Path],
    shard_plan: Union[str, Path],
    shard_id: str,
) -> RemoteInputs:
    """Load and validate one shard without consulting submit-host paths."""
    config = _load_object(resolved_config, "resolved config")
    workload = _load_object(resolved_workload, "resolved workload")
    ownership = _load_object(shard_plan, "shard plan")
    if ownership.get("schema_version") != 1:
        raise ConfigurationError("unsupported AOne shard plan schema_version")
    expected_config = ownership.get("config_fingerprint")
    expected_workload = ownership.get("workload_fingerprint")
    if _fingerprint(config) != expected_config:
        raise ConfigurationError("AOne resolved config fingerprint mismatch")
    if _fingerprint(workload) != expected_workload:
        raise ConfigurationError("AOne resolved workload fingerprint mismatch")

    shards = ownership.get("shards")
    if not isinstance(shards, list):
        raise ConfigurationError("AOne shard plan shards must be an array")
    selected: Mapping[str, Any] | None = None
    seen_ids = set()
    for item in shards:
        if not isinstance(item, Mapping) or not isinstance(item.get("shard_id"), str):
            raise ConfigurationError("AOne shard plan contains an invalid shard")
        current = item["shard_id"]
        if current in seen_ids:
            raise ConfigurationError(f"duplicate AOne shard ID: {current}")
        seen_ids.add(current)
        if current == shard_id:
            selected = item
    if selected is None:
        raise ConfigurationError(f"unknown AOne shard ID: {shard_id}")
    raw_cases = selected.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigurationError(f"AOne shard {shard_id} has no owned cases")
    cases = []
    case_ids = []
    for item in raw_cases:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            raise ConfigurationError(f"AOne shard {shard_id} contains an invalid case")
        cases.append(dict(item))
        case_ids.append(item["case_id"])
    if len(set(case_ids)) != len(case_ids):
        raise ConfigurationError(f"AOne shard {shard_id} contains duplicate cases")

    combined = dict(config)
    combined["workload"] = workload
    run_id = ownership.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ConfigurationError("AOne shard plan run_id must be a non-empty string")
    return RemoteInputs(
        run_id=run_id,
        shard_id=shard_id,
        config_fingerprint=expected_config,
        workload_fingerprint=expected_workload,
        config=combined,
        owned_case_ids=tuple(case_ids),
        owned_cases=tuple(cases),
    )


def _default_executor(inputs: RemoteInputs):
    from rtp_llm.test.perf_test.execution.local import _default_session_factory
    from rtp_llm.test.perf_test.modes import PrefillAdapter
    from rtp_llm.test.perf_test.prefill.probe import (
        probe_cache_block,
        resolve_probe_block_size,
    )
    from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
    from rtp_llm.test.perf_test.test_util import _load_tokenizer

    runtime_inputs = resolve_remote_runtime_inputs(inputs)
    model = runtime_inputs.config["model"]
    tokenizer = _load_tokenizer(
        model["tokenizer"],
        checkpoint_path=model["checkpoint"],
        model_type=model["type"],
    )
    commit_tail = int(inputs.config["workload"]["commit_tail_tokens"])
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            tokenizer,
            commit_tail_tokens=commit_tail,
            cache_block_size=resolve_probe_block_size(inputs.config),
        ),
        probe=lambda context: probe_cache_block(
            context.session.client,
            tokenizer,
            resolve_probe_block_size(inputs.config),
            commit_tail,
        ),
    )
    executor = RemoteShardExecutor(
        session_factory=_default_session_factory, adapter=adapter
    )
    return _BoundRemoteExecutor(executor, runtime_inputs)


class _BoundRemoteExecutor:
    def __init__(self, executor: Any, inputs: RemoteInputs):
        self._executor = executor
        self._inputs = inputs

    def execute(self, _inputs: RemoteInputs, output_directory: Path):
        return self._executor.execute(self._inputs, output_directory)


def _write_setup_failure(
    inputs: RemoteInputs, output_directory: Path, error: Exception
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_directory / "shard_result.json",
        {
            "schema_version": 1,
            "run_id": inputs.run_id,
            "shard_id": inputs.shard_id,
            "mode": "prefill",
            "status": "failed",
            "config_fingerprint": inputs.config_fingerprint,
            "workload_fingerprint": inputs.workload_fingerprint,
            "owned_case_ids": list(inputs.owned_case_ids),
            "completed_case_ids": [],
            "cases": [],
            "engine_diagnostics": {
                "log_path": None,
                "timeline_files": [],
                "coredumps": {},
                "exit_code": None,
                "collection_errors": [f"setup: {type(error).__name__}: {error}"],
            },
            "failure": {"type": type(error).__name__, "message": str(error)},
        },
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    executor_factory: Optional[Callable[[RemoteInputs], Any]] = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--resolved-workload", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    loaded = load_remote_inputs(
        resolved_config=args.resolved_config,
        resolved_workload=args.resolved_workload,
        shard_plan=args.shard_plan,
        shard_id=args.shard_id,
    )
    output_dir = args.output_dir
    if output_dir is None:
        undeclared = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if not undeclared:
            raise ConfigurationError(
                "AOne worker requires --output-dir or TEST_UNDECLARED_OUTPUTS_DIR"
            )
        output_dir = str(Path(undeclared) / f"perf_test_{loaded.shard_id}")
    factory = _default_executor if executor_factory is None else executor_factory
    output_path = Path(output_dir)
    try:
        result = factory(loaded).execute(loaded, output_path)
        exit_code = int(result.exit_code)
    except Exception as error:
        _write_setup_failure(loaded, output_path, error)
        exit_code = 5
    print(
        json.dumps(
            {
                "run_id": loaded.run_id,
                "shard_id": loaded.shard_id,
                "case_ids": loaded.owned_case_ids,
                "output_dir": output_dir,
                "exit_code": exit_code,
            },
            sort_keys=True,
        )
    )
    return exit_code


SHARD_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RemoteShardResult:
    run_id: str
    shard_id: str
    state: RunState
    config_fingerprint: str
    workload_fingerprint: str
    owned_case_ids: Tuple[str, ...]
    completed_case_ids: Tuple[str, ...]
    cases: Tuple[Mapping[str, Any], ...]
    engine_diagnostics: EngineDiagnostics
    failure: Optional[Dict[str, str]] = None

    @property
    def exit_code(self) -> int:
        if self.state == RunState.COMPLETED:
            return 0
        if self.state == RunState.INTERRUPTED:
            return 130
        return 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SHARD_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "shard_id": self.shard_id,
            "mode": "prefill",
            "status": self.state.value,
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "owned_case_ids": list(self.owned_case_ids),
            "completed_case_ids": list(self.completed_case_ids),
            "cases": [dict(case) for case in self.cases],
            "engine_diagnostics": self.engine_diagnostics.to_dict(),
            "failure": None if self.failure is None else dict(self.failure),
        }


def _plan_for_shard(inputs: RemoteInputs, output: Path) -> ExecutionPlan:
    execution = inputs.config.get("execution")
    model = inputs.config.get("model")
    if not isinstance(execution, Mapping) or execution.get("backend") != "aone":
        raise ValueError("remote shard requires normalized AOne execution config")
    if not isinstance(model, Mapping) or not isinstance(model.get("topology"), Mapping):
        raise ValueError("remote shard requires normalized model topology")
    topology = model["topology"]
    cards = topology["tp"] * topology["dp"] * topology["pp"]
    resource = AOneResourcePlan(
        backend="aone",
        resource=execution["resource"],
        gpu_type=execution["gpu_type"],
        gpus_per_shard=cards,
        shard_count=execution["shard_count"],
        max_parallel=execution["max_parallel"],
    )
    cases = []
    for item in inputs.owned_cases:
        identity = item.get("identity")
        current_id = item.get("case_id")
        if not isinstance(identity, Mapping) or current_id != case_id(
            "prefill", identity
        ):
            raise ValueError("remote shard case identity does not match its case ID")
        cases.append(PlannedCase(current_id, freeze(identity)))
    return ExecutionPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        mode="prefill",
        action="run",
        backend="aone",
        resolved_config=freeze(inputs.config),
        config_fingerprint=inputs.config_fingerprint,
        workload_fingerprint=inputs.workload_fingerprint,
        cases=tuple(cases),
        resource_plan=resource,
        result_directory=str(output),
    )


def _failure(error: BaseException) -> Dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


class RemoteShardExecutor:
    def __init__(
        self, *, session_factory: Callable[[ExecutionPlan], Any], adapter: Any
    ):
        self._session_factory = session_factory
        self._adapter = adapter

    def execute(
        self, inputs: RemoteInputs, output_directory: Union[str, Path]
    ) -> RemoteShardResult:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        (output / "results").mkdir()
        (output / "results" / "cases.jsonl").touch()
        plan = _plan_for_shard(inputs, output)
        store = RunStore(output)
        session = self._session_factory(plan)
        state = RunState.FAILED
        failure: Optional[Dict[str, str]] = None
        try:
            with session as active_session:
                mode_result = self._adapter(
                    ModeExecutionContext(
                        plan=plan,
                        session=active_session,
                        store=store,
                    )
                )
                state = mode_result.state
                if mode_result.failure is not None:
                    failure = _failure(mode_result.failure)
        except EngineInterrupted as error:
            state = RunState.INTERRUPTED
            failure = _failure(error)
        except Exception as error:
            state = RunState.FAILED
            failure = _failure(error)

        committed = store.read_prefill_cases(plan)
        completed = tuple(
            case_id for case_id in inputs.owned_case_ids if case_id in committed
        )
        cases = tuple(committed[case_id] for case_id in completed)
        result = RemoteShardResult(
            run_id=inputs.run_id,
            shard_id=inputs.shard_id,
            state=state,
            config_fingerprint=inputs.config_fingerprint,
            workload_fingerprint=inputs.workload_fingerprint,
            owned_case_ids=inputs.owned_case_ids,
            completed_case_ids=completed,
            cases=cases,
            engine_diagnostics=session.diagnostics,
            failure=failure,
        )
        atomic_write_json(output / "shard_result.json", result.to_dict())
        return result


if __name__ == "__main__":
    raise SystemExit(main())
