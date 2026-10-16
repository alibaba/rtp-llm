"""RTP-LLM batch decode performance test — main entry point.

Three-phase flow:
  1. Configure — parse args, resolve paths, build PerfTestConfig
  2. Serve    — start engine, query engine status, print config tables
  3. Run      — dispatch to prefill or decode runner, collect timelines
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.cache_grid.config.grid_config import (
    _dedupe_cache_grid_cases,
    _load_cache_grid_cases,
    _resolve_cache_block_size,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import profile_environment
from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
    CacheGridRunner,
    MaterializedCaseStore,
    PrefixPromptFactory,
    is_grouped_case,
    resume_config_fingerprint,
    validate_cache_grid_resume,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    grid_token_budget,
    grid_workspace_tokens,
    validate_token_budget,
    workspace_capacity,
)
from rtp_llm.test.perf_test.dataclass import PerfTestConfig
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.distribution_runner import DistributionRunner
from rtp_llm.test.perf_test.grid_runner import GridRunner
from rtp_llm.test.perf_test.perf_config import (
    _apply_engine_env,
    _apply_run_overrides,
    _engine_arg_argv,
    _parse_name_value,
    _replace_cli_value,
    parse_args,
    prepare_config,
    resolve_perf_engine_paths,
)
from rtp_llm.test.perf_test.perf_utils import (
    _is_sensitive_name,
    _redact_argv,
    _sanitize_nested_provenance,
    _sanitize_provenance_value,
    collect_timeline_files,
    filter_bs_by_kvcache,
    print_config_table,
    query_engine_status,
    write_test_info,
)
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.perf_test.tps_runner import TpsBinarySearchRunner
from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps

__all__ = [
    "_engine_arg_argv",
    "_parse_name_value",
    "_redact_argv",
    "main",
    "parse_args",
    "run_single",
]

# ---------------------------------------------------------------------------
#  Backward-compatible wrapper (used by external callers)
# ---------------------------------------------------------------------------


def run_single(
    port: int,
    dp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 20,
    tp_size: int = 1,
    generate_config: Optional[Dict[str, Any]] = None,
):
    return GridRunner(
        port,
        dp_size,
        batch_size_list,
        input_len_list,
        input_query_dict,
        is_decode=is_decode,
        dump_json_path=dump_json_path,
        decode_test_length=decode_test_length,
        tp_size=tp_size,
        generate_config=generate_config,
    ).run()


# ---------------------------------------------------------------------------
#  Grid-mode helpers
# ---------------------------------------------------------------------------


_REPRODUCTION_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_WORLD_SIZE",
    "START_PORT",
    "TOKENIZERS_PARALLELISM",
    "WORLD_SIZE",
}
_REPRODUCTION_ENV_PREFIXES = (
    "CACHE_",
    "DG_JIT_",
    "DSV4_",
    "PERF_",
    "PREFILL_",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TILELANG_",
    "TRITON_",
)


def _capture_reproduction_env(
    engine_env_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Capture performance-relevant environment values with credential redaction."""
    requested = set(engine_env_names or [])
    requested.update(
        name
        for name in os.environ
        if name in _REPRODUCTION_ENV_NAMES
        or name.startswith(_REPRODUCTION_ENV_PREFIXES)
    )
    captured = {}
    for name in sorted(requested):
        if name not in os.environ:
            continue
        captured[name] = _sanitize_provenance_value(
            name, os.environ[name], allow_plaintext=True
        )
    return captured


def _build_cache_resume_config(
    args: argparse.Namespace,
    remaining_args: List[str],
    engine_env_names: Optional[List[str]],
    expected_cache_block_size: int,
) -> Dict[str, Any]:
    """Build the stable model/workload config guarded across resumed attempts."""
    return {
        "schema_version": 1,
        "model": {
            "model_type": extract_arg(remaining_args, "model_type"),
            "checkpoint_path": extract_arg(remaining_args, "checkpoint_path"),
            "tokenizer_path": extract_arg(remaining_args, "tokenizer_path"),
        },
        "engine": {
            "args": _redact_argv(remaining_args),
            "environment": _capture_reproduction_env(engine_env_names),
            "dp_size": args.dp_size,
            "max_seq_len": args.max_seq_len,
            "concurrency_limit": args.concurrency_limit,
        },
        "workload": {
            "partial": args.partial,
            "decode_test_length": args.decode_test_length,
            "cache_measure_runs": args.cache_measure_runs,
            "cache_skip_reuse_validation": args.cache_skip_reuse_validation,
            "cache_commit_tail_tokens": args.cache_commit_tail_tokens,
            "expected_cache_block_size": expected_cache_block_size,
            "cache_request_transport": args.cache_request_transport,
            **(
                {"workspace_tokens": args.cache_workspace_tokens}
                if getattr(args, "cache_workspace_tokens", None) is not None
                else {}
            ),
        },
    }


def _configure_cache_batch_limits(args, remaining, cases, *, cache_alignment=None):
    """Configure token capacity without changing profile admission or context limits."""
    batch_size = max(c["batch_size"] for c in cases)
    if args.concurrency_limit < batch_size:
        raise ValueError(
            f"--concurrency_limit={args.concurrency_limit} cannot admit cache-grid "
            f"batch_size={batch_size}; increase it to at least {batch_size}"
        )
    workspace = getattr(args, "cache_workspace_tokens", None)
    if workspace is not None:
        if type(cache_alignment) is not int or cache_alignment <= 0:
            raise ValueError("fixed workspace requires grid.generator.cache_alignment")
        if args.dp_size != 1:
            raise ValueError("fixed token workspace requires dp_size=1")
        tp = int(extract_arg(remaining, "tp_size") or 1)
        cp_method = extract_arg(remaining, "cp_rotate_method") or "DISABLED"
        if cp_method not in (
            "DISABLED",
            "ALL_GATHER",
            "ALL_GATHER_WITH_OVERLAP",
            "ALLTOALL",
        ):
            raise ValueError("unsupported CP method for fixed token workspace")
        cp_alignment = 2 * tp if cp_method != "DISABLED" and tp > 1 else 1
        if tp <= 0 or cache_alignment % cp_alignment:
            raise ValueError(
                "cache_alignment must be a multiple of the CP execution alignment"
            )
        if args.cache_shared_seed:
            raise ValueError("fixed workspace does not support cache_shared_seed mode")
        budget = int(extract_arg(remaining, "max_batch_tokens_size") or workspace)
        budget = min(budget or workspace, workspace)
        stats = validate_token_budget(
            cases,
            workspace_tokens=workspace,
            commit_tail=args.cache_commit_tail_tokens,
            block=cache_alignment,
            output_tokens=args.decode_test_length,
            token_budget=budget,
        )
        args.max_seq_len = workspace_capacity(workspace, cache_alignment)
        limits = {
            "max_batch_tokens_size": budget,
        }
        index, cleaned = 0, []
        while index < len(remaining):
            item = remaining[index]
            name = item.split("=", 1)[0].lstrip("-")
            if name in limits:
                index += 1 if "=" in item else 2
            else:
                cleaned.append(item)
                index += 1
        remaining[:] = cleaned + [f"--{key}={value}" for key, value in limits.items()]
        logging.info(
            "cache fixed token workspace: max_seq_len=%d capacity=%s",
            args.max_seq_len,
            stats,
        )
        return
    if batch_size <= 1:
        return
    if args.dp_size != 1:
        raise ValueError("fixed cache batches currently require dp_size=1")
    total_tokens = max(
        sum(
            g["count"] * g["input_len"]
            for g in c.get(
                "request_groups",
                [{"count": c["batch_size"], "input_len": c["input_len"]}],
            )
        )
        for c in cases
    )
    for name, required in (("max_batch_tokens_size", total_tokens),):
        current = int(extract_arg(remaining, name) or 0)
        if current >= required:
            continue
        flag = "--" + name
        cleaned = []
        index = 0
        while index < len(remaining):
            value = remaining[index]
            if value == flag:
                index += 2
            elif value.startswith(flag + "="):
                index += 1
            else:
                cleaned.append(value)
                index += 1
        remaining[:] = cleaned + [flag, str(required)]
        logging.info(
            "cache fixed batch: raised %s from %d to %d", name, current, required
        )


def _load_materialized_case_store(
    root: str,
    cases: List[Dict[str, int]],
    measure_runs: int,
    grid_sha256: str,
    profile_sha256: Optional[str] = None,
) -> MaterializedCaseStore:
    """Load a precomputed prompt store and prove it matches this grid."""
    store = MaterializedCaseStore(root)
    stored = store.load_cases()
    if store.run_count != measure_runs:
        raise ValueError(
            f"case store {root} was materialized with run_count="
            f"{store.run_count}, but --cache_measure_runs={measure_runs}"
        )
    if store.grid_sha256 and store.grid_sha256 != grid_sha256:
        raise ValueError(
            f"case store {root} was materialized from a different grid JSON "
            f"(sha256 {store.grid_sha256[:12]} != {grid_sha256[:12]}); "
            "rerun --materialize_cache_cases"
        )
    if store.profile_sha256 != (profile_sha256 or ""):
        raise ValueError(
            f"case store {root} was materialized from a different profile "
            f"(sha256 {store.profile_sha256[:12]} != {(profile_sha256 or '')[:12]}); "
            "rerun --materialize_cache_cases"
        )

    def geometry(case: Dict[str, int]) -> tuple:
        return (
            case["case_id"],
            CacheGridRunner.case_key(case),
            case["batch_size"],
            case["input_len"],
            case["cache_len"],
        )

    if [geometry(c) for c in stored] != [geometry(c) for c in cases]:
        raise ValueError(
            f"case store {root} holds {len(stored)} cases but this run plans "
            f"{len(cases)}; the grid JSON or block-size dedup changed since "
            "materialization. Rerun --materialize_cache_cases."
        )
    return store


def _finalize_completed_cache_checkpoint(
    result_dir: str, checkpoint: Dict[str, Any], cases: List[Dict[str, Any]]
) -> None:
    """Atomically mark an all-successful interrupted checkpoint as complete."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    total = len(cases)
    finalized = dict(checkpoint)
    finalized.update(
        complete=True,
        status="completed",
        updated_at=now,
        total_cases=total,
        completed_cases=total,
        progress={
            "total_cases": total,
            "attempted_cases": total,
            "completed_cases": total,
            "failed_or_invalid_cases": 0,
            "pending_cases": 0,
            "progress_pct": 100.0,
            "last_completed_case": (
                CacheGridRunner._case_descriptor(cases[-1]) if cases else None
            ),
            "next_pending_case": None,
        },
    )
    root = Path(result_dir)
    result_path = root / "cache_grid_results.json"
    tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp.write_text(json.dumps(finalized, indent=2), encoding="utf-8")
    os.replace(tmp, result_path)
    progress = {
        "schema_version": 1,
        "mode": "prefix_cache_grid_progress",
        "updated_at": now,
        "resume_count": finalized.get("resume_count", 0),
        "run_config_sha256": finalized.get("run_config_sha256"),
        "progress": finalized["progress"],
    }
    progress_path = root / "cache_grid_progress.json"
    progress_tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    progress_tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(progress_tmp, progress_path)


def _write_test_info(
    args: argparse.Namespace,
    remaining_args: List[str],
    engine_env_names: Optional[List[str]] = None,
    status: str = "completed",
    expected_cache_block_size: int = 0,
    resume_config: Optional[Dict[str, Any]] = None,
    effective_cache_grpc_port: Optional[int] = None,
) -> None:
    """Persist a reproducible, credential-safe test configuration."""
    model_type = extract_arg(remaining_args, "model_type") or os.environ.get(
        "MODEL_TYPE"
    )
    checkpoint_path = extract_arg(remaining_args, "checkpoint_path") or os.environ.get(
        "CHECKPOINT_PATH"
    )
    profile = getattr(args, "_profile", None)
    profile_sha256 = getattr(args, "_profile_sha256", None)
    path = os.path.join(args.result_dir, "test_info.json")
    previous: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, ValueError):
            logging.warning("Ignoring unreadable previous test info at %s", path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    requested_cache_grpc_port = (
        (args.cache_grpc_port or None) if args.cache_grid_json else None
    )
    attempt_count = int(previous.get("attempt_count", 0))
    if status == "running":
        attempt_count += 1
    info = {
        "status": status,
        "started_at": previous.get("started_at", now),
        "updated_at": now,
        "last_attempt_started_at": (
            now if status == "running" else previous.get("last_attempt_started_at", now)
        ),
        "attempt_count": attempt_count,
        "model_type": _sanitize_provenance_value(
            "model_type", model_type, allow_plaintext=True
        ),
        "checkpoint_path": _sanitize_provenance_value(
            "checkpoint_path", checkpoint_path, allow_plaintext=True
        ),
        "tp_size": extract_arg(remaining_args, "tp_size", "1"),
        "dp_size": args.dp_size,
        "cache_grid_json": args.cache_grid_json or None,
        "cache_grpc_port_requested": requested_cache_grpc_port,
        "cache_grpc_port_effective": (
            effective_cache_grpc_port if args.cache_grid_json else None
        ),
        "resume_config_sha256": resume_config_fingerprint(resume_config),
        "profile_sha256": profile_sha256,
    }
    launch_path = os.path.join(args.result_dir, "cache_perf_launch.json")
    if args.cache_grid_json and os.path.isfile(launch_path):
        # The unified launcher already owns the reproducible configuration.
        # Keep results' run_config as the authoritative effective config used
        # by the resume guard; do not repeat it or the raw argv in test_info.
        with open(launch_path, encoding="utf-8") as stream:
            launch = json.load(stream)
        if launch.get("schema_version") != 2:
            raise ValueError("unsupported launch manifest")
        info.update(
            schema_version=4,
            config_files={
                "launch": "cache_perf_launch.json",
                "profile": launch.get("profile_file", "profile.snapshot.json"),
                "grid": "grid.snapshot.json",
            },
            config_file_sha256={
                "cache_perf_launch.json": hashlib.sha256(
                    Path(launch_path).read_bytes()
                ).hexdigest(),
                **launch.get("snapshots", {}),
            },
        )
    else:
        tokenizer_path = extract_arg(
            remaining_args, "tokenizer_path"
        ) or os.environ.get("TOKENIZER_PATH")
        info.update(
            {
                "schema_version": 3,
                "tokenizer_path": _sanitize_provenance_value(
                    "tokenizer_path", tokenizer_path, allow_plaintext=True
                ),
                "max_seq_len": args.max_seq_len,
                "concurrency_limit": args.concurrency_limit,
                "decode_test_length": args.decode_test_length,
                "seq_size_per_block": extract_arg(
                    remaining_args, "seq_size_per_block", None
                ),
                "cache_measure_runs": (
                    args.cache_measure_runs if args.cache_grid_json else None
                ),
                "cache_skip_reuse_validation": (
                    args.cache_skip_reuse_validation if args.cache_grid_json else None
                ),
                "cache_request_timeout": (
                    args.cache_request_timeout if args.cache_grid_json else None
                ),
                "cache_commit_tail_tokens": (
                    args.cache_commit_tail_tokens if args.cache_grid_json else None
                ),
                "partial": args.partial,
                "warmup_runs": int(os.environ.get("PERF_FORMAL_WARMUP_RUNS", "1")),
                "measure_runs": int(os.environ.get("PERF_MEASURE_RUNS", "1")),
                "profile_runs": int(os.environ.get("PERF_PROFILE_RUNS", "1")),
                "expected_cache_block_size": (
                    expected_cache_block_size if args.cache_grid_json else None
                ),
                "cache_case_store": args.cache_case_files or None,
                "cache_profile_runs": args.cache_profile_runs,
                "cache_profile_case_ids": args.cache_profile_case_ids,
                "cache_profile_only": args.cache_profile_only,
                "cache_profile_trace_timeout": args.cache_profile_trace_timeout,
                "cache_profile_backend": args.cache_profile_backend,
                "cache_nsys_session": args.cache_nsys_session,
                "cache_nsys_path": args.cache_nsys_path,
                "cache_nsys_tail_seconds": args.cache_nsys_tail_seconds,
                "cache_request_transport": (
                    args.cache_request_transport if args.cache_grid_json else None
                ),
                "cache_grpc_port": (
                    effective_cache_grpc_port
                    if effective_cache_grpc_port is not None
                    else requested_cache_grpc_port
                ),
                "dataset_name": args.dataset_name or None,
                "dataset_path": _sanitize_provenance_value(
                    "dataset_path",
                    args.dataset_path or args.dataset or None,
                    allow_plaintext=True,
                ),
                "engine_args": _redact_argv(remaining_args),
                "engine_env_names": sorted(engine_env_names or []),
                "engine_environment": _capture_reproduction_env(engine_env_names),
                "argv": _redact_argv(sys.argv),
                "resume_config": _sanitize_nested_provenance(
                    "resume_config", resume_config
                ),
                "profile": _sanitize_nested_provenance("profile", profile),
            }
        )
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as stream:
        json.dump(info, stream, indent=2)
    os.replace(tmp_path, path)
    logging.info("Wrote test info to %s", path)


def _ensure_xgrammar_lib_path() -> None:
    import sys

    so_name = "libxgrammar_bindings.so"
    for p in sys.path:
        xgrammar_dir = os.path.join(p, "xgrammar")
        if os.path.isfile(os.path.join(xgrammar_dir, so_name)):
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if xgrammar_dir not in current.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{xgrammar_dir}:{current}" if current else xgrammar_dir
                )
                logging.info(f"Added {xgrammar_dir} to LD_LIBRARY_PATH")
            return


def _ensure_batch_decode_scheduler(remaining: List[str]) -> None:
    """Require the scheduler used by fixed-batch cache-grid measurements."""
    value = extract_arg(remaining, "use_batch_decode_scheduler")
    if value is None:
        remaining.extend(["--use_batch_decode_scheduler", "1"])
        return
    if str(value).lower() not in ("1", "true"):
        raise ValueError(
            "cache-grid measurements require --use_batch_decode_scheduler=1"
        )


def _prepare_cache_profile_result_dir(args):
    if not args.cache_profile_only:
        return
    if not args.cache_profile_flat_output:
        args.result_dir = str(
            Path(args.result_dir) / "cache_profile_replays" / uuid.uuid4().hex
        )
        return
    directory = Path(args.result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    allowed = {"cache_perf_launch.json", "profile.snapshot.json", "grid.snapshot.json"}
    if any(p.name not in allowed or not p.is_file() for p in directory.iterdir()):
        raise ValueError("flat profile output requires a fresh isolated directory")
    # Exclusive claim prevents concurrent/repeated invocations from overwriting results.
    with (directory / ".cache_profile_started").open("x") as stream:
        stream.write("profile-only isolated output\n")


def _effective_grid_max_seq_len(
    args: argparse.Namespace, input_len_list: List[int]
) -> int:
    """Grid-mode max_seq_len: decode headroom, but never below explicit --max_seq_len.

    prepare_config() sizes grid max_seq_len as max(input_len) + decode_test_length.
    DSv4 perf targets size the KV pool from an explicitly larger --max_seq_len
    (e.g. --input_len 65536 --decode_test_length 100 --max_seq_len 65664), so that
    request must win.  Distribution mode already does this in prepare_config().
    """
    needed_seq_len = max(input_len_list) + args.decode_test_length
    return max(needed_seq_len, args.max_seq_len)


def _require_cache_grid_success(metrics: List[Dict[str, Any]]) -> None:
    """Fail the command after the runner has checkpointed every non-ok case."""
    failed = [metric for metric in metrics if metric.get("status") != "ok"]
    if not failed:
        return
    counts: Dict[str, int] = {}
    for metric in failed:
        status = str(metric.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    raise RuntimeError(
        f"cache grid failed {len(failed)}/{len(metrics)} cases ({summary}); "
        "see cache_grid_results.json"
    )


def _explicit_batch_size_list(args: argparse.Namespace) -> Optional[List[int]]:
    """--batch_size as given on the command line, or None when it was defaulted."""
    batch_size_explicit = getattr(
        args,
        "batch_size_explicit",
        any(a.startswith("--batch_size") for a in sys.argv[1:]),
    )
    if not batch_size_explicit:
        return None
    return [int(x) for x in args.batch_size.split(",")]


_PERFORMANCE_ENV_NAMES = {
    "ACT_TYPE",
    "CACHE_CONFIG",
    "CACHE_STORE_TYPE",
    "CHECKPOINT_PATH",
    "CONCURRENCY_LIMIT",
    "CP_ROTATE_METHOD",
    "DEVICE_NAME",
    "DEVICE_RESERVE_MEMORY_BYTES",
    "DP_SIZE",
    "DSV4_CHUNK_TOKENS",
    "DSV4_FIXED_POOL_BLOCKS",
    "ENABLE_CUDA_GRAPH",
    "EP_SIZE",
    "FP8_KV_CACHE",
    "GEN_NUM_PER_CYCLE",
    "INT8_MODE",
    "KV_CACHE_MEM_BYTES",
    "KV_CACHE_MEM_MB",
    "LOAD_METHOD",
    "LOCAL_WORLD_SIZE",
    "MAX_BATCH_SIZE",
    "MAX_BATCH_TOKENS_SIZE",
    "MAX_CONTEXT_BATCH_SIZE",
    "MAX_SEQ_LEN",
    "MODEL_TYPE",
    "PREFILL_CP_KV_CACHE_SHARDED",
    "QUANTIZATION",
    "RESERVER_RUNTIME_MEM_MB",
    "SEQ_SIZE_PER_BLOCK",
    "SP_ACT_TYPE",
    "SP_CHECKPOINT_PATH",
    "SP_MODEL_TYPE",
    "SP_TYPE",
    "TOKENIZER_PATH",
    "TP_SIZE",
    "USE_DEEPEP_LOW_LATENCY",
    "USE_DEEPEP_MOE",
    "WORLD_SIZE",
}
_PERFORMANCE_ENV_PREFIXES = (
    "CACHE_",
    "CUDA_",
    "DEEP_EP_",
    "DG_JIT_",
    "DSV4_",
    "ENABLE_",
    "FP8_",
    "GEN_TIMELINE_",
    "INT8_",
    "KV_CACHE_",
    "LOAD_",
    "MODEL_",
    "MOE_",
    "NCCL_",
    "PERF_",
    "PREFILL_",
    "QUANTIZATION_",
    "RTP_LLM_",
    "SP_",
    "TORCH_",
    "USE_DEEP",
)


def _is_performance_runtime_name(name: str, explicit_names: set[str]) -> bool:
    return (
        name in explicit_names
        or name in _PERFORMANCE_ENV_NAMES
        or name.startswith(_PERFORMANCE_ENV_PREFIXES)
    )


def _fingerprint_engine_env(names: List[str]) -> Dict[str, str]:
    """Capture runtime env with plaintext limited to the reviewed safe allowlist."""
    explicit_names = set(names)
    relevant_names = {
        name
        for name in os.environ
        if _is_performance_runtime_name(name, explicit_names)
        and not _is_sensitive_name(name)
    }
    return {
        name: str(
            _sanitize_provenance_value(
                name,
                os.environ[name],
                allow_plaintext=name in _PERFORMANCE_ENV_NAMES,
            )
        )
        for name in sorted(relevant_names)
    }


def _effective_performance_config(
    engine_args: List[str],
    engine_env_names: List[str],
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Resolve effective runtime values with CLI taking precedence over env."""
    values = _fingerprint_engine_env(engine_env_names)
    explicit_names = set(engine_env_names)
    sources = {
        name: "engine_env" if name in explicit_names else "environment"
        for name in values
    }
    index = 0
    while index < len(engine_args):
        argument = engine_args[index]
        index += 1
        if not argument.startswith("--"):
            continue
        option = argument[2:]
        if "=" in option:
            option, value = option.split("=", 1)
        elif index < len(engine_args) and not engine_args[index].startswith("--"):
            value = engine_args[index]
            index += 1
        else:
            value = "1"
        name = option.replace("-", "_").upper()
        if not _is_performance_runtime_name(name, explicit_names):
            continue
        if _is_sensitive_name(name):
            values.pop(name, None)
            sources.pop(name, None)
            continue
        values[name] = str(
            _sanitize_provenance_value(
                name,
                value,
                allow_plaintext=name in _PERFORMANCE_ENV_NAMES,
            )
        )
        sources[name] = "cli"

    for name, value in (cli_overrides or {}).items():
        normalized = name.replace("-", "_").upper()
        if _is_sensitive_name(normalized):
            continue
        values[normalized] = str(
            _sanitize_provenance_value(
                normalized,
                value,
                allow_plaintext=normalized in _PERFORMANCE_ENV_NAMES,
            )
        )
        sources[normalized] = "cli"
    return {
        "values": dict(sorted(values.items())),
        "sources": dict(sorted(sources.items())),
    }


# ---------------------------------------------------------------------------
#  Phase 3: Run — prefill / decode dispatch
# ---------------------------------------------------------------------------


def _run_prefill(
    port: int,
    dp_size: int,
    config: PerfTestConfig,
    input_query_dict: Dict[int, str],
    batch_size_list: Optional[List[int]] = None,
    **kwargs: Any,
) -> None:
    """Prefill grid run.

    Defaults to BS=1 (prefill measures single-request TTFT); prepare_config()
    pins config.batch_size_list to [1] for --partial 2 as well.  DSv4 prefill
    targets (e.g. v4_flash_cp4_ep4_prefill_64k_perf) sweep prefill at
    batch_size > 1, so main() forwards an explicitly requested --batch_size.
    """
    if not config.input_len_list:
        return
    GridRunner(
        port,
        dp_size,
        batch_size_list or [1],
        config.input_len_list,
        input_query_dict,
        is_decode=False,
        **kwargs,
    ).run()


def _run_decode(
    port: int,
    dp_size: int,
    args: argparse.Namespace,
    config: PerfTestConfig,
    input_query_dict: Dict[int, str],
    engine_status: Dict[str, Any],
    **kwargs: Any,
) -> None:
    max_kv = (
        float(engine_status.get("max_kv_tokens", float("inf")))
        if engine_status
        else float("inf")
    )

    if args.target_tpot > 0:
        runner = TpsBinarySearchRunner(
            port,
            dp_size,
            args.target_tpot,
            max_bs=args.concurrency_limit,
            **kwargs,
        )
        if config.is_distribution:
            assert config.test_config is not None
            runner.run_distribution(config.test_config, input_query_dict)
        else:
            max_bs_per_len = {
                il: max(
                    [bs for bs in config.batch_size_list if bs * il <= max_kv] or [1]
                )
                for il in config.input_len_list
            }
            runner.run_grid(config.input_len_list, input_query_dict, max_bs_per_len)
    else:
        if config.is_distribution:
            assert config.test_config is not None
            DistributionRunner(
                port,
                dp_size,
                config.test_config,
                input_query_dict,
                **kwargs,
            ).run()
        else:
            for input_len in config.input_len_list:
                filtered_bs = filter_bs_by_kvcache(
                    config.batch_size_list, input_len, max_kv
                )
                if not filtered_bs:
                    logging.warning(
                        f"No BS fits KV cache for input_len={input_len}, skipping"
                    )
                    continue
                GridRunner(
                    port,
                    dp_size,
                    filtered_bs,
                    [input_len],
                    input_query_dict,
                    is_decode=True,
                    **kwargs,
                ).run()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()
    args, remaining = parse_args()
    engine_env_names = _apply_engine_env(
        [
            f"{key}={value}"
            for key, value in profile_environment(args._profile or {}).items()
        ]
    )
    engine_env_names = sorted(
        set(engine_env_names + _apply_engine_env(args.engine_env))
    )
    _ensure_xgrammar_lib_path()
    _apply_run_overrides(args)
    remaining = resolve_perf_engine_paths(remaining)
    _ensure_batch_decode_scheduler(remaining)
    generate_config = json.loads(args.generate_config)
    if args.cache_profile_runs and args.dp_size != 1:
        raise ValueError(
            "cache-grid profiling currently requires DP=1; all TP ranks are captured"
        )
    _prepare_cache_profile_result_dir(args)
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)

    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {_redact_argv(remaining)}")

    if args.cache_grid_json:
        if args.partial != 2:
            raise ValueError("--cache_grid_json is prefill-only; use --partial=2")
        if args.cache_measure_runs <= 0:
            raise ValueError("--cache_measure_runs must be positive")
        if args.cache_request_timeout <= 0:
            raise ValueError("--cache_request_timeout must be positive")
        if args.cache_commit_tail_tokens <= 0:
            raise ValueError("--cache_commit_tail_tokens must be positive")
        if args.cache_checkpoint_every <= 0:
            raise ValueError("--cache_checkpoint_every must be positive")
        if args.materialize_cache_cases and args.cache_case_files:
            raise ValueError(
                "--materialize_cache_cases and --cache_case_files are mutually "
                "exclusive"
            )

        cases = _load_cache_grid_cases(args.cache_grid_json)
        with open(args.cache_grid_json, "rb") as stream:
            grid_bytes = stream.read()
        grid_payload = json.loads(grid_bytes)
        workspace = grid_workspace_tokens(grid_payload)
        if workspace is not None:
            if args.cache_workspace_tokens not in (None, workspace):
                raise ValueError("CLI/grid workspace_tokens mismatch")
            args.cache_workspace_tokens = workspace
            budget = str(min(workspace, grid_token_budget(grid_payload)))
            if extract_arg(remaining, "max_batch_tokens_size") is None:
                remaining.append("--max_batch_tokens_size=" + budget)
            else:
                _replace_cli_value(remaining, "max_batch_tokens_size", budget)
        expected_block_size = _resolve_cache_block_size(grid_payload)
        _configure_cache_batch_limits(
            args, remaining, cases, cache_alignment=expected_block_size
        )
        grid_metadata = {
            key: grid_payload.get(key)
            for key in ("schema_version", "kind", "generator", "summary")
            if key in grid_payload
        }
        grid_sha256 = hashlib.sha256(grid_bytes).hexdigest()
        if expected_block_size > 0:
            deduped = _dedupe_cache_grid_cases(cases, expected_block_size)
            if len(deduped) < len(cases):
                logging.warning(
                    "cache grid: dropped %d of %d cases that collapse onto the same "
                    "%d-token physical block bucket",
                    len(cases) - len(deduped),
                    len(cases),
                    expected_block_size,
                )
                cases = deduped
        logging.info(
            "cache grid plan: cases=%d sha256=%s expected_block_size=%d metadata=%s",
            len(cases),
            grid_sha256,
            expected_block_size,
            grid_metadata,
        )
        if args.cache_shared_seed:
            if args.cache_request_transport != "dashsc_input_ids":
                raise ValueError("--cache_shared_seed requires dashsc_input_ids")
            if any(is_grouped_case(c) or c["batch_size"] != 1 for c in cases):
                raise ValueError(
                    "--cache_shared_seed supports only ungrouped batch_size=1"
                )
            if (
                args.cache_profile_runs
                or args.cache_profile_only
                or args.cache_case_files
                or args.materialize_cache_cases
            ):
                raise ValueError(
                    "--cache_shared_seed does not support profiling or materialized cases"
                )
            if (
                expected_block_size <= 0
                or args.cache_commit_tail_tokens < expected_block_size
            ):
                raise ValueError(
                    "--cache_shared_seed requires a known block size and full commit tail"
                )
        unknown_profile_ids = set(args.cache_profile_case_ids) - {
            int(c["case_id"]) for c in cases
        }
        if unknown_profile_ids:
            raise ValueError(
                f"unknown/deduplicated cache profile case IDs: {sorted(unknown_profile_ids)}"
            )
        if any(
            is_grouped_case(c) and c["case_id"] in args.cache_profile_case_ids
            for c in cases
        ):
            raise ValueError(
                "cache profiling currently supports only ungrouped batch_size=1 cases"
            )
        if args.cache_profile_runs and args.materialize_cache_cases:
            raise ValueError(
                "cache profiling cannot be combined with --materialize_cache_cases"
            )
        resume_config = _build_cache_resume_config(
            args, remaining, engine_env_names, expected_block_size
        )
        if args.cache_shared_seed:
            resume_config["cache_seed_mode"] = "shared_prefix_v1"
        checkpoint = validate_cache_grid_resume(
            args.result_dir,
            grid_sha256=grid_sha256,
            profile_sha256=getattr(args, "_profile_sha256", None),
            measure_runs=args.cache_measure_runs,
            cache_commit_tail_tokens=args.cache_commit_tail_tokens,
            expected_block_size=expected_block_size,
            request_transport=args.cache_request_transport,
            run_config=resume_config,
            shared_seed=args.cache_shared_seed,
            allow_resume_mismatch=args.allow_resume_mismatch,
            require_resume=args.require_cache_resume,
        )
        if checkpoint is not None:
            logging.info(
                "cache grid: preflight resume accepted progress=%s",
                checkpoint.get("progress"),
            )
        _write_test_info(
            args,
            remaining,
            engine_env_names,
            status="running",
            expected_cache_block_size=expected_block_size,
            resume_config=resume_config,
        )
        checkpoint_ok_keys = (
            {
                str(row.get("case_key"))
                for row in checkpoint.get("metrics", [])
                if isinstance(row, dict) and row.get("status") == "ok"
            }
            if checkpoint is not None
            else set()
        )
        planned_case_keys = {CacheGridRunner.case_key(case) for case in cases}
        if (
            checkpoint is not None
            and checkpoint_ok_keys == planned_case_keys
            and not args.cache_profile_runs
            and not args.materialize_cache_cases
        ):
            if checkpoint.get("complete") is not True:
                _finalize_completed_cache_checkpoint(args.result_dir, checkpoint, cases)
            logging.info(
                "cache grid: checkpoint already contains all %d successful cases; "
                "skipping tokenizer and model startup",
                len(cases),
            )
            _write_test_info(
                args,
                remaining,
                engine_env_names,
                status="completed",
                expected_cache_block_size=expected_block_size,
                resume_config=resume_config,
            )
            return args.result_dir
        for case in [group for c in cases for group in c.get("request_groups", [c])]:
            cache_len = int(case["cache_len"])
            input_len = int(case["input_len"])
            if (
                "count" in case
                and expected_block_size
                and cache_len % expected_block_size
            ):
                raise ValueError(
                    f"cache-grid cache_len must align to block size {expected_block_size}: {case}"
                )
            if cache_len and cache_len % args.cache_commit_tail_tokens:
                raise ValueError(
                    "cache-grid cache_len must align to "
                    f"--cache_commit_tail_tokens={args.cache_commit_tail_tokens}: "
                    f"{case}"
                )
            if cache_len and cache_len + args.cache_commit_tail_tokens > input_len:
                raise ValueError(
                    "cache-grid cache_len must leave one commit tail before "
                    f"server startup: {case}"
                )
        max_input_len = max(int(case["input_len"]) for case in cases)
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        if not tokenizer_path:
            raise ValueError(
                "cache-grid mode requires --tokenizer_path or --checkpoint_path"
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        if args.materialize_cache_cases:
            store = MaterializedCaseStore(args.materialize_cache_cases)
            stats = store.materialize(
                cases,
                PrefixPromptFactory(tokenizer),
                args.cache_measure_runs,
                grid_metadata=grid_metadata,
                grid_sha256=grid_sha256,
                profile_sha256=getattr(args, "_profile_sha256", "") or "",
            )
            logging.info(
                "materialized %d cache-grid cases (%d cached) to %s: %s",
                stats["cases"],
                stats["cached_cases"],
                args.materialize_cache_cases,
                stats,
            )
            _write_test_info(
                args,
                remaining,
                engine_env_names,
                status="completed",
                expected_cache_block_size=expected_block_size,
                resume_config=resume_config,
            )
            return args.result_dir

        case_store = None
        if args.cache_case_files:
            case_store = _load_materialized_case_store(
                args.cache_case_files,
                cases,
                args.cache_measure_runs,
                grid_sha256,
                getattr(args, "_profile_sha256", None),
            )
            logging.info(
                "cache grid: using materialized case store %s", args.cache_case_files
            )

        server = EngineServer(args, remaining)
        effective_cache_grpc_port = None
        try:
            server.start(
                max_seq_len=max(
                    max_input_len + args.decode_test_length, args.max_seq_len
                ),
                # Admission must allow the complete fixed batch to reach the scheduler.
                max_concurrency=args.concurrency_limit,
            )
            if args.cache_request_transport == "dashsc_input_ids":
                effective_cache_grpc_port = args.cache_grpc_port or server.port + 8
            # CacheGridRunner bypasses GridRunner/BatchPerfImpl, whose run() normally
            # switches BatchDecodeScheduler to prefill. Configure it explicitly before
            # the block-size probe, cache seeds, or measurement requests are issued.
            server.set_scheduler_mode(batch_size=1, mode="prefill")
            metrics = CacheGridRunner(
                server.port,
                tokenizer,
                cases,
                args.result_dir,
                request_timeout=args.cache_request_timeout,
                shared_seed=args.cache_shared_seed,
                measure_runs=args.cache_measure_runs,
                checkpoint_every=args.cache_checkpoint_every,
                cache_commit_tail_tokens=args.cache_commit_tail_tokens,
                skip_reuse_validation=args.cache_skip_reuse_validation,
                grid_metadata=grid_metadata,
                grid_sha256=grid_sha256,
                expected_block_size=expected_block_size,
                case_store=case_store,
                profile=getattr(args, "_profile", None),
                profile_sha256=getattr(args, "_profile_sha256", None),
                allow_resume_mismatch=args.allow_resume_mismatch,
                require_resume=args.require_cache_resume,
                request_transport=args.cache_request_transport,
                grpc_port=effective_cache_grpc_port,
                run_config=resume_config,
                profile_runs=args.cache_profile_runs,
                profile_case_ids=args.cache_profile_case_ids,
                profile_only=args.cache_profile_only,
                profile_flat_output=args.cache_profile_flat_output,
                profile_tp_size=int(
                    extract_arg(remaining, "tp_size") or os.environ.get("TP_SIZE", "1")
                ),
                profile_trace_timeout=args.cache_profile_trace_timeout,
                profile_backend=args.cache_profile_backend,
                nsys_path=args.cache_nsys_path,
                nsys_session=args.cache_nsys_session,
                nsys_tail_seconds=args.cache_nsys_tail_seconds,
            ).run()
            _require_cache_grid_success(metrics)
            collect_timeline_files(args.result_dir)
        except BaseException:
            _write_test_info(
                args,
                remaining,
                engine_env_names,
                status="failed",
                expected_cache_block_size=expected_block_size,
                resume_config=resume_config,
                effective_cache_grpc_port=effective_cache_grpc_port,
            )
            raise
        else:
            _write_test_info(
                args,
                remaining,
                engine_env_names,
                status="completed",
                expected_cache_block_size=expected_block_size,
                resume_config=resume_config,
                effective_cache_grpc_port=effective_cache_grpc_port,
            )
            return args.result_dir
        finally:
            try:
                server.stop()
            finally:
                summarize_and_cleanup_coredumps(args.result_dir)

    # Phase 1: Configure
    config = prepare_config(args, remaining)
    if not config.is_distribution:
        config.max_seq_len = _effective_grid_max_seq_len(args, config.input_len_list)
    effective_runtime_config = _effective_performance_config(
        remaining,
        engine_env_names,
        {
            "DP_SIZE": args.dp_size,
            "MAX_SEQ_LEN": config.max_seq_len,
            "CONCURRENCY_LIMIT": config.max_concurrency,
        },
    )
    write_test_info(
        args,
        remaining,
        engine_env_names,
        status="running",
        effective_max_seq_len=config.max_seq_len,
        service_concurrency_limit=config.max_concurrency,
        effective_runtime_config=effective_runtime_config,
    )

    # Phase 2: Serve
    server = EngineServer(args, remaining)
    try:
        server.start(
            max_seq_len=config.max_seq_len,
            max_concurrency=config.max_concurrency,
            use_batch_decode_scheduler=True,
        )
        engine_status = query_engine_status(server.port)
        print_config_table(args, config, engine_status, remaining)

        # Phase 3: Run
        input_query_dict = create_query(input_len_list=config.all_seq_lens)
        runner_kwargs = dict(
            dump_json_path=args.result_dir,
            decode_test_length=args.decode_test_length,
            generate_config=generate_config,
            num_measures=args.num_measures,
            log_path=server.log_file_path or "",
        )

        if args.partial == 2:
            _run_prefill(
                server.port,
                args.dp_size,
                config,
                input_query_dict,
                batch_size_list=_explicit_batch_size_list(args),
                **runner_kwargs,
            )

        if args.partial == 1:
            _run_decode(
                server.port,
                args.dp_size,
                args,
                config,
                input_query_dict,
                engine_status,
                **runner_kwargs,
            )

        # Cleanup
        collect_timeline_files(args.result_dir)
        server.stop()
        write_test_info(
            args,
            remaining,
            engine_env_names,
            status="completed",
            effective_max_seq_len=config.max_seq_len,
            service_concurrency_limit=config.max_concurrency,
            effective_runtime_config=effective_runtime_config,
        )

        if args.partial != 2:
            from rtp_llm.test.perf_test.visualization import plot_decode_results

            try:
                plot_decode_results(args.result_dir)
            except Exception as e:
                logging.warning(f"plot_decode_results failed: {e}")
    finally:
        summarize_and_cleanup_coredumps(args.result_dir)

    return args.result_dir


if __name__ == "__main__":
    main()
