"""Utility functions for perf test: BS generation, KV cache filtering, engine status query, timeline collection."""

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from rtp_llm.test.perf_test.dataclass import PerfTestConfig


def auto_generate_bs_list(concurrency_limit: int) -> List[int]:
    """Auto-generate BS list: <64 step=8, >=64 step=64."""
    bs_list = [1]
    bs = 8
    while bs <= concurrency_limit:
        bs_list.append(bs)
        if bs < 64:
            bs += 8
        else:
            bs += 64
    return bs_list


def filter_bs_by_kvcache(
    bs_list: List[int], input_len: int, max_kv_tokens_per_dp: float
) -> List[int]:
    """Filter BS list by KV cache capacity. BS is per-DP."""
    return [bs for bs in bs_list if bs * input_len <= max_kv_tokens_per_dp]


def query_engine_status(port: int) -> Dict[str, Any]:
    """Query /cache_status and /worker_status, return unified engine info.

    Returns dict with keys: max_kv_tokens, total_kv_cache, block_size,
    concurrency_limit, dp_size.  Returns empty dict on failure.
    """
    result: Dict[str, Any] = {}
    try:
        cache = requests.get(f"http://127.0.0.1:{port}/cache_status", timeout=10).json()
        worker = requests.get(
            f"http://127.0.0.1:{port}/worker_status", timeout=10
        ).json()
        if "error" in cache or "error" in worker:
            logging.warning(f"Engine status error: cache={cache}, worker={worker}")
            return result

        cache_results = cache.get("results", [cache])
        per_dp_kv = [
            int(c.get("total_kv_cache", 0)) * int(c.get("block_size", 1))
            for c in cache_results
        ]
        result["max_kv_tokens"] = min(per_dp_kv) if per_dp_kv else 0
        result["total_kv_cache"] = int(cache.get("total_kv_cache", 0))
        result["block_size"] = int(cache.get("block_size", 1))
        result["dp_size"] = int(cache.get("dp_size", 1))
        result["concurrency_limit"] = int(worker.get("frontend_concurrency_limit", 0))
    except Exception as e:
        logging.warning(f"Failed to query engine status: {e}")
    return result


def collect_timeline_files(result_dir: str) -> None:
    """Collect profiler timeline JSON files into a timelines/ subdirectory."""
    time.sleep(3)
    timeline_dir = os.path.join(result_dir, "timelines")
    pattern = os.path.join(result_dir, "*.json")
    timeline_files = [
        f
        for f in glob.glob(pattern)
        if os.path.basename(f).startswith(("profiler_ts", "profiler_"))
        or "_wr" in os.path.basename(f)
    ]
    if timeline_files:
        os.makedirs(timeline_dir, exist_ok=True)
        for f in timeline_files:
            dst = os.path.join(timeline_dir, os.path.basename(f))
            shutil.move(f, dst)
            logging.debug(f"Collected timeline: {dst}")
    else:
        logging.info("No timeline files found in %s", result_dir)


def _is_sensitive_name(name: str) -> bool:
    """Identify credential fields without hiding semantic token-count settings."""
    normalized = name.lstrip("-").lower().replace("-", "_")
    return (
        any(
            marker in normalized
            for marker in (
                "password",
                "passwd",
                "secret",
                "access_key",
                "api_key",
                "apikey",
                "credential",
                "private_key",
                "sso_empid_hash",
                "authorization",
            )
        )
        or normalized == "token"
        or normalized.endswith("_token")
    )


_SAFE_PERSISTED_ARG_NAMES = {
    "act_type",
    "batch_size",
    "cache_commit_tail_tokens",
    "cache_grid_json",
    "cache_measure_runs",
    "cache_request_timeout",
    "checkpoint_path",
    "concurrency_limit",
    "cp_rotate_method",
    "dataset",
    "dataset_name",
    "dataset_path",
    "decode_test_length",
    "device_name",
    "dp_size",
    "dsv4_chunk_tokens",
    "dsv4_fixed_pool_blocks",
    "enable_cuda_graph",
    "ep_size",
    "fp8_kv_cache",
    "input_len",
    "int8_mode",
    "kv_cache_mem_bytes",
    "kv_cache_mem_mb",
    "load_method",
    "max_batch_size",
    "max_batch_tokens_size",
    "max_context_batch_size",
    "max_context_tokens",
    "max_seq_len",
    "measure_runs",
    "model_type",
    "num_measures",
    "partial",
    "profile_runs",
    "quantization",
    "reserver_runtime_mem_mb",
    "result_dir",
    "seq_size_per_block",
    "sp_act_type",
    "sp_checkpoint_path",
    "sp_model_type",
    "sp_type",
    "target_tpot",
    "test_json",
    "tokenizer_path",
    "tp_size",
    "use_batch_decode_scheduler",
    "use_deepep_low_latency",
    "use_deepep_moe",
    "warmup_runs",
    "world_size",
}
_HASHED_VALUE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?i)(?:^|[?#&;,\s])(?:authorization|auth|password|passwd|pwd|secret|token|"
    r"key|access[_-]?token|access[_-]?key|account[_-]?key|api[_-]?key|"
    r"client[_-]?secret|credential|signature|sig|awsaccesskeyid|googleaccessid|"
    r"x-amz-[^=;,&\s]+)\s*[=:]"
)
_URI_USERINFO_RE = re.compile(r"://[^/@\s]+@", re.IGNORECASE)
_URI_QUERY_RE = re.compile(r"[a-z][a-z0-9+.-]*:[^\s]*\?", re.IGNORECASE)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+")


def _hash_provenance_value(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value_contains_credentials(value: str) -> bool:
    return bool(
        _URI_USERINFO_RE.search(value)
        or _URI_QUERY_RE.search(value)
        or _CREDENTIAL_VALUE_RE.search(value)
        or _AUTH_VALUE_RE.search(value)
    )


def _sanitize_provenance_value(
    name: str, value: Any, *, allow_plaintext: bool
) -> str | None:
    """Return a deterministic persisted value without credential-bearing text."""
    if value is None:
        return None
    text = str(value)
    if text == "***" or _HASHED_VALUE_RE.fullmatch(text):
        return text
    if _is_sensitive_name(name):
        return "***"
    if not allow_plaintext or _value_contains_credentials(text):
        return _hash_provenance_value(text)
    return text


def _sanitize_assignment(value: str, *, inline: bool) -> str:
    if "=" not in value:
        return _hash_provenance_value(value)
    name, raw_value = value.split("=", 1)
    sanitized = _sanitize_provenance_value(
        name,
        raw_value,
        allow_plaintext=name.lower().replace("-", "_") in _SAFE_PERSISTED_ARG_NAMES,
    )
    if inline and sanitized == "***":
        return "***"
    return f"{name}={sanitized}"


def _redact_argv(argv: List[str]) -> List[str]:
    """Persist safe argv values; redact credentials and hash unknown values."""
    redacted: List[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        index += 1
        if not item.startswith("--"):
            redacted.append(_hash_provenance_value(item))
            continue

        option = item[2:]
        if "=" in option:
            name, raw_value = option.split("=", 1)
            if name.lower().replace("-", "_") in ("engine_arg", "engine_env"):
                sanitized = _sanitize_assignment(raw_value, inline=True)
            else:
                sanitized = _sanitize_provenance_value(
                    name,
                    raw_value,
                    allow_plaintext=name.lower().replace("-", "_")
                    in _SAFE_PERSISTED_ARG_NAMES,
                )
            redacted.append(f"--{name}={sanitized}")
            continue

        name = option
        redacted.append(item)
        if index >= len(argv) or argv[index].startswith("--"):
            continue
        raw_value = argv[index]
        index += 1
        if name.lower().replace("-", "_") in ("engine_arg", "engine_env"):
            redacted.append(_sanitize_assignment(raw_value, inline=False))
        else:
            sanitized = _sanitize_provenance_value(
                name,
                raw_value,
                allow_plaintext=name.lower().replace("-", "_")
                in _SAFE_PERSISTED_ARG_NAMES,
            )
            redacted.append(str(sanitized))
    return redacted


def write_test_info(
    args: argparse.Namespace,
    remaining_args: List[str],
    engine_env_names: Optional[List[str]] = None,
    status: str = "completed",
    effective_max_seq_len: Optional[int] = None,
    service_concurrency_limit: Optional[int] = None,
    effective_runtime_config: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """Persist a reproducible, credential-safe test configuration."""
    from rtp_llm.test.perf_test.dataset import extract_arg

    model_type = extract_arg(remaining_args, "model_type") or os.environ.get(
        "MODEL_TYPE"
    )
    checkpoint_path = extract_arg(remaining_args, "checkpoint_path") or os.environ.get(
        "CHECKPOINT_PATH"
    )
    tokenizer_path = extract_arg(remaining_args, "tokenizer_path") or os.environ.get(
        "TOKENIZER_PATH"
    )
    model_type = _sanitize_provenance_value(
        "model_type", model_type, allow_plaintext=True
    )
    checkpoint_path = _sanitize_provenance_value(
        "checkpoint_path", checkpoint_path, allow_plaintext=True
    )
    tokenizer_path = _sanitize_provenance_value(
        "tokenizer_path", tokenizer_path, allow_plaintext=True
    )
    runtime_config = effective_runtime_config or {"values": {}, "sources": {}}
    runtime_config = {
        **runtime_config,
        "values": {
            name: _sanitize_provenance_value(
                name,
                value,
                allow_plaintext=name.lower().replace("-", "_")
                in _SAFE_PERSISTED_ARG_NAMES,
            )
            for name, value in runtime_config.get("values", {}).items()
        },
    }
    requested_max_seq_len = int(args.max_seq_len)
    service_max_seq_len = int(
        effective_max_seq_len
        if effective_max_seq_len is not None
        else requested_max_seq_len
    )
    requested_concurrency_limit = int(args.concurrency_limit)
    actual_concurrency_limit = int(
        service_concurrency_limit
        if service_concurrency_limit is not None
        else requested_concurrency_limit
    )
    info = {
        "schema_version": 5,
        "status": status,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "tp_size": extract_arg(remaining_args, "tp_size", "1"),
        "dp_size": args.dp_size,
        "max_seq_len": service_max_seq_len,
        "requested_max_seq_len": requested_max_seq_len,
        "effective_max_seq_len": service_max_seq_len,
        "concurrency_limit": requested_concurrency_limit,
        "requested_concurrency_limit": requested_concurrency_limit,
        "service_concurrency_limit": actual_concurrency_limit,
        "decode_test_length": args.decode_test_length,
        "cache_grid_json": _sanitize_provenance_value(
            "cache_grid_json", args.cache_grid_json or None, allow_plaintext=True
        ),
        "cache_measure_runs": (
            args.cache_measure_runs if args.cache_grid_json else None
        ),
        "cache_request_timeout": (
            args.cache_request_timeout if args.cache_grid_json else None
        ),
        "cache_commit_tail_tokens": (
            args.cache_commit_tail_tokens if args.cache_grid_json else None
        ),
        "partial": args.partial,
        "warmup_runs": int(os.environ.get("PERF_FORMAL_WARMUP_RUNS", "1")),
        "measure_runs": int(
            os.environ.get("PERF_MEASURE_RUNS", str(getattr(args, "num_measures", 1)))
        ),
        "profile_runs": int(os.environ.get("PERF_PROFILE_RUNS", "1")),
        "dataset_name": _sanitize_provenance_value(
            "dataset_name", args.dataset_name or None, allow_plaintext=True
        ),
        "dataset_path": _sanitize_provenance_value(
            "dataset_path",
            args.dataset_path or args.dataset or None,
            allow_plaintext=True,
        ),
        "engine_args": _redact_argv(remaining_args),
        "engine_env_names": sorted(engine_env_names or []),
        "effective_runtime_config": runtime_config,
        "argv": _redact_argv(sys.argv),
    }
    path = os.path.join(args.result_dir, "test_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Wrote test info to {path}")


def print_config_table(
    args: argparse.Namespace,
    config: PerfTestConfig,
    engine_status: Dict[str, Any],
    remaining: List[str],
) -> None:
    """Print test config, engine info, and per-input_len filtered BS."""
    from prettytable import PrettyTable

    from rtp_llm.test.perf_test.dataset import extract_arg

    mode = "distribution" if config.is_distribution else "grid"
    if args.target_tpot > 0:
        mode += "+tps"

    model_table = PrettyTable()
    model_table.title = "Model & Engine Info"
    model_table.field_names = ["Parameter", "Value"]
    model_table.align = "l"
    model_table.add_row(
        [
            "Model Type",
            os.environ.get("MODEL_TYPE", extract_arg(remaining, "model_type", "N/A")),
        ]
    )
    model_table.add_row(
        [
            "Checkpoint",
            os.environ.get(
                "CHECKPOINT_PATH", extract_arg(remaining, "checkpoint_path", "N/A")
            ),
        ]
    )
    model_table.add_row(["TP Size", extract_arg(remaining, "tp_size", "1")])
    model_table.add_row(["DP Size", args.dp_size])
    if engine_status:
        model_table.add_row(
            ["KV Cache Tokens (per DP)", engine_status.get("max_kv_tokens", "N/A")]
        )
        model_table.add_row(
            ["KV Cache Blocks (per DP)", engine_status.get("total_kv_cache", "N/A")]
        )
        model_table.add_row(["Block Size", engine_status.get("block_size", "N/A")])
        model_table.add_row(
            [
                "Concurrency Limit",
                engine_status.get("concurrency_limit", args.concurrency_limit),
            ]
        )
    logging.info("Model & engine info:\n" + str(model_table))

    table = PrettyTable()
    table.title = "Perf Test Configuration"
    table.field_names = ["Parameter", "Value"]
    table.align = "l"
    table.add_row(["Mode", mode])
    table.add_row(["Target TPOT (ms)", args.target_tpot or "N/A"])
    table.add_row(["Batch Sizes", config.batch_size_list or "N/A"])
    table.add_row(["Input Lengths", config.input_len_list or "N/A"])
    table.add_row(["Decode Test Length", args.decode_test_length])
    table.add_row(["Partial", args.partial])
    table.add_row(["Generate Config", args.generate_config])
    logging.info("Test configuration:\n" + str(table))

    if config.input_len_list and not config.is_distribution and args.partial != 2:
        max_kv = (
            float(engine_status.get("max_kv_tokens", float("inf")))
            if engine_status
            else float("inf")
        )
        bs_table = PrettyTable()
        bs_table.title = "Effective BS per Input Length (after KV cache filter)"
        bs_table.field_names = ["Input Len", "Max BS (KV cache)", "Candidate BS List"]
        bs_table.align = "l"
        for il in config.input_len_list:
            filtered = filter_bs_by_kvcache(config.batch_size_list, il, max_kv)
            max_bs = filtered[-1] if filtered else 0
            bs_table.add_row([il, max_bs, filtered or "N/A"])
        logging.info("Effective BS per input length:\n" + str(bs_table))
