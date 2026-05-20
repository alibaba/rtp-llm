#!/usr/bin/env python3
"""RTP-vLLM precision gate helper.

This CLI intentionally does not start model services. Service deployment depends
on GPU assignment and local model paths; see the sibling SKILL.md for launch
templates. The CLI handles the repeatable precision-gate pieces:

- health-check RTP/vLLM endpoints
- run RTP natural self-roll through the existing repeat runner
- run an explicit RTP teacher-forced gate after the decode service is started
  with RTP_TEACHER_FORCE_* environment variables
- create RTP teacher-force token files from a saved vLLM oracle
- compare generated_ids.json files against a saved vLLM oracle
- report first token diff, hashes, and repetition patterns
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


DEFAULT_RUN_ROOT = Path(
    "/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959"
)
DEFAULT_VLLM_ORACLE_IDS = (
    DEFAULT_RUN_ROOT
    / "outputs/vllm_stable_nodump_ignoreeos_len1000_oracle_20260517_220909_record89_20260517_220916"
    / "vllm_run01/generated_ids.json"
)
DEFAULT_RTP_FINAL_IDS = (
    DEFAULT_RUN_ROOT
    / "outputs/rtp_final_novllm_selfroll_len1000_20260518_101414_record89_20260518_101422"
    / "rtp_run01/generated_ids.json"
)
EXPECTED_RECORD89_HASH1000 = "986b77c92c844fc6"


def load_ids(path: Path) -> List[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path} does not contain a JSON list")
    return [int(x) for x in value]


def write_teacher_force_tokens(vllm_ids_path: Path, output_path: Path) -> Dict[str, Any]:
    ids = load_ids(vllm_ids_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(x) for x in ids) + "\n", encoding="utf-8")
    return {
        "vllm_ids": str(vllm_ids_path),
        "teacher_force_tokens": str(output_path),
        "num_tokens": len(ids),
        "hash": short_hash(ids),
        "head": ids[:16],
        "tail": ids[-16:],
    }


def short_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(str(list(ids)).encode("utf-8")).hexdigest()[:16]


def first_diff(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None


def longest_same_token_run(ids: Sequence[int]) -> Dict[str, Optional[int]]:
    if not ids:
        return {"start": None, "run_len": 0, "token_id": None}
    best_start, best_len, best_id = 0, 0, int(ids[0])
    i = 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        if j - i > best_len:
            best_start, best_len, best_id = i, j - i, int(ids[i])
        i = j
    return {"start": best_start, "run_len": best_len, "token_id": best_id}


def tail_period(ids: Sequence[int], max_period: int) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[int, int, int, List[int]]] = None
    for period_len in range(1, max_period + 1):
        if period_len > len(ids):
            break
        unit = [int(x) for x in ids[-period_len:]]
        repeats = 1
        while (repeats + 1) * period_len <= len(ids):
            start = -(repeats + 1) * period_len
            end = -repeats * period_len
            if [int(x) for x in ids[start:end]] != unit:
                break
            repeats += 1
        if repeats >= 2:
            candidate = (repeats * period_len, period_len, repeats, unit)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    total, period_len, repeats, unit = best
    return {
        "total_tokens": total,
        "period_len": period_len,
        "repeats": repeats,
        "period": unit,
    }


def compare_ids(
    rtp_ids_path: Path,
    vllm_ids_path: Path,
    prefix_len: Optional[int],
    max_period: int,
) -> Dict[str, Any]:
    rtp_ids = load_ids(rtp_ids_path)
    vllm_ids = load_ids(vllm_ids_path)
    compare_len = prefix_len if prefix_len is not None else min(len(rtp_ids), len(vllm_ids))
    rtp_prefix = rtp_ids[:compare_len]
    vllm_prefix = vllm_ids[:compare_len]
    diff = first_diff(rtp_prefix, vllm_prefix)
    result: Dict[str, Any] = {
        "rtp_ids": str(rtp_ids_path),
        "vllm_ids": str(vllm_ids_path),
        "rtp_len": len(rtp_ids),
        "vllm_len": len(vllm_ids),
        "compare_len": compare_len,
        "first_diff": diff,
        "equal_prefix": diff is None and len(rtp_prefix) == len(vllm_prefix),
        "rtp_hash": short_hash(rtp_prefix),
        "vllm_hash": short_hash(vllm_prefix),
        "rtp_extra_after_compare": rtp_ids[compare_len : compare_len + 16],
        "vllm_extra_after_compare": vllm_ids[compare_len : compare_len + 16],
        "rtp_longest_same_token_run": longest_same_token_run(rtp_prefix),
        "vllm_longest_same_token_run": longest_same_token_run(vllm_prefix),
        "rtp_tail_period": tail_period(rtp_prefix, max_period),
        "vllm_tail_period": tail_period(vllm_prefix, max_period),
    }
    if diff is not None:
        lo = max(0, diff - 16)
        hi = min(compare_len, diff + 32)
        result["diff_window"] = {
            "start": lo,
            "end": hi,
            "rtp": rtp_prefix[lo:hi],
            "vllm": vllm_prefix[lo:hi],
        }
    return result


def print_json(result: Dict[str, Any], json_out: Optional[Path]) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")


def check_health(url: str, timeout: float) -> Dict[str, Any]:
    health_url = url.rstrip("/") + "/health"
    start = time.time()
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            body = resp.read(256).decode("utf-8", errors="replace")
            return {
                "url": health_url,
                "ok": 200 <= int(resp.status) < 300,
                "status": int(resp.status),
                "elapsed_sec": round(time.time() - start, 3),
                "body": body,
            }
    except Exception as exc:  # noqa: BLE001 - CLI should report any endpoint failure.
        return {
            "url": health_url,
            "ok": False,
            "status": None,
            "elapsed_sec": round(time.time() - start, 3),
            "error": repr(exc),
        }


def default_runner() -> Path:
    candidates = [
        Path("/tmp/repeat_stability_from_q_ignore_eos.py"),
        Path("/data3/dsv4_repeat_compare/scripts/repeat_stability_from_q_ignore_eos.py"),
        Path("/data3/dsv4_repeat_compare/scripts/repeat_stability_from_q.py"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find repeat_stability_from_q runner; pass --runner explicitly"
    )


def newest_run_dir(out_root: Path, name: str, record_index: int, since: float) -> Path:
    pattern = f"{name}_record{record_index}_*"
    candidates = [
        p
        for p in out_root.glob(pattern)
        if p.is_dir() and p.stat().st_mtime >= since - 1
    ]
    if not candidates:
        raise FileNotFoundError(f"No run dir matching {out_root / pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_rtp_selfroll(args: argparse.Namespace) -> Path:
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runner = args.runner.expanduser().resolve() if args.runner else default_runner()
    name = args.name or f"rtp_selfroll_record{args.record_index}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable,
        str(runner),
        "--q-path",
        str(args.q_path),
        "--backend",
        "rtp",
        "--record-index",
        str(args.record_index),
        "--repeats",
        str(args.repeats),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--stop-repeat-run",
        str(args.stop_repeat_run),
        "--top-k",
        str(args.top_k),
        "--rtp-mode",
        args.rtp_mode,
        "--rtp-url",
        args.rtp_url,
        "--timeout",
        str(args.timeout),
        "--out-root",
        str(out_root),
        "--name",
        name,
    ]
    if args.grpc_addr:
        cmd.extend(["--grpc-addr", args.grpc_addr])

    print("[run]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    role_ports = infer_role_addr_ports(args)
    env.update({key: str(value) for key, value in role_ports.items()})
    print("[run-env]", json.dumps(role_ports, sort_keys=True), flush=True)
    start = time.time()
    subprocess.run(cmd, check=True, env=env)
    return newest_run_dir(out_root, name, args.record_index, start)


def _port_from_url(url: str) -> Optional[int]:
    parsed = urlparse(url)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _port_from_hostport(value: str) -> Optional[int]:
    if not value:
        return None
    tail = value.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def infer_role_addr_ports(args: argparse.Namespace) -> Dict[str, int]:
    prefill_http = args.prefill_http_port or _port_from_url(args.prefill_url)
    prefill_grpc = args.prefill_grpc_port or (
        prefill_http + 1 if prefill_http is not None else None
    )
    decode_grpc = args.decode_grpc_port or _port_from_hostport(args.grpc_addr)
    decode_http = args.decode_http_port
    if decode_http is None and decode_grpc is not None:
        decode_http = decode_grpc - 1
    if decode_http is None:
        decode_http = _port_from_url(args.rtp_url)
    if decode_grpc is None and decode_http is not None:
        decode_grpc = decode_http + 1
    missing = [
        name
        for name, value in (
            ("prefill_http", prefill_http),
            ("prefill_grpc", prefill_grpc),
            ("decode_http", decode_http),
            ("decode_grpc", decode_grpc),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"cannot infer role_addrs ports: {missing}")
    return {
        "RTP_PREFILL_HTTP_PORT": int(prefill_http),
        "RTP_PREFILL_GRPC_PORT": int(prefill_grpc),
        "RTP_DECODE_HTTP_PORT": int(decode_http),
        "RTP_DECODE_GRPC_PORT": int(decode_grpc),
    }


def generated_id_files(run_dir: Path) -> List[Path]:
    files = sorted(run_dir.glob("rtp_run*/generated_ids.json"))
    if not files and (run_dir / "generated_ids.json").exists():
        files = [run_dir / "generated_ids.json"]
    if not files:
        raise FileNotFoundError(f"No generated_ids.json found under {run_dir}")
    return files


def command_compare(args: argparse.Namespace) -> int:
    result = compare_ids(args.rtp_ids, args.vllm_ids, args.prefix_len, args.max_period)
    if args.expect_hash:
        result["expected_hash"] = args.expect_hash
        result["hash_matches"] = (
            result["rtp_hash"] == args.expect_hash
            and result["vllm_hash"] == args.expect_hash
        )
    print_json(result, args.json_out)
    if args.fail_on_diff and not result["equal_prefix"]:
        return 2
    if args.expect_hash and not result.get("hash_matches", False):
        return 3
    return 0


def command_health(args: argparse.Namespace) -> int:
    checks = [check_health(url, args.timeout) for url in args.urls]
    result = {"checks": checks, "all_ok": all(item["ok"] for item in checks)}
    print_json(result, args.json_out)
    return 0 if result["all_ok"] else 2


def command_check_process_libs(args: argparse.Namespace) -> int:
    worktree = args.rtp_worktree.expanduser().resolve()
    bazel_bin = (worktree / "bazel-bin").resolve()
    expected = {
        "libth_transformer.so",
        "librtp_compute_ops.so",
        "libth_transformer_config.so",
    }
    maps = Path(f"/proc/{int(args.pid)}/maps")
    if not maps.exists():
        raise FileNotFoundError(f"{maps} does not exist")

    loaded: Dict[str, List[str]] = {name: [] for name in expected}
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        for name in expected:
            if name in line:
                path = line.rsplit(maxsplit=1)[-1]
                if path not in loaded[name]:
                    loaded[name].append(path)

    result: Dict[str, Any] = {
        "pid": int(args.pid),
        "worktree": str(worktree),
        "bazel_bin": str(bazel_bin),
        "libs": loaded,
        "missing": sorted(name for name, paths in loaded.items() if not paths),
        "wrong_root": {},
    }
    wrong_root: Dict[str, List[str]] = {}
    for name, paths in loaded.items():
        bad = [path for path in paths if not Path(path).resolve().is_relative_to(bazel_bin)]
        if bad:
            wrong_root[name] = bad
    result["wrong_root"] = wrong_root
    result["ok"] = not result["missing"] and not wrong_root
    print_json(result, args.json_out)
    return 0 if result["ok"] else 2


def command_check_env_dump(args: argparse.Namespace) -> int:
    dump_dir = args.dump_dir.expanduser().resolve()
    if not dump_dir.exists():
        raise FileNotFoundError(f"{dump_dir} does not exist")

    expected_pairs: Dict[str, str] = {}
    for item in args.expect:
        if "=" not in item:
            raise ValueError(f"--expect must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        expected_pairs[key] = value

    files = sorted(dump_dir.glob("*.json"))
    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        stage = str(data.get("stage", ""))
        if args.stage and stage != args.stage:
            continue
        env = data.get("env", {})
        entry = {
            "path": str(path),
            "stage": stage,
            "pid": data.get("pid"),
            "ppid": data.get("ppid"),
            "env": env,
        }
        entries.append(entry)

        missing = [key for key in args.require_key if key not in env]
        mismatched = {
            key: {"actual": env.get(key), "expected": value}
            for key, value in expected_pairs.items()
            if env.get(key) != value
        }
        if missing or mismatched:
            failures.append(
                {
                    "path": str(path),
                    "stage": stage,
                    "pid": data.get("pid"),
                    "missing": missing,
                    "mismatched": mismatched,
                }
            )

    result = {
        "dump_dir": str(dump_dir),
        "stage_filter": args.stage,
        "expect": expected_pairs,
        "require_key": args.require_key,
        "entries": entries,
        "failures": failures,
        "ok": bool(entries) and not failures,
    }
    print_json(result, args.json_out)
    return 0 if result["ok"] else 2


def command_teacher_force_file(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    result = write_teacher_force_tokens(args.vllm_ids.expanduser().resolve(), output)
    result["offset"] = int(args.offset)
    result["env"] = {
        "RTP_TEACHER_FORCE_TOKENS": str(output),
        "RTP_TEACHER_FORCE_OFFSET": str(int(args.offset)),
    }
    result["note"] = (
        "RTP teacher forcing is read by the decode service process at startup. "
        "Restart RTP decode with these env vars; setting them only on the client "
        "runner does not affect an already-running service."
    )
    print_json(result, args.json_out)
    return 0


def compare_run_dir(args: argparse.Namespace, run_dir: Path) -> int:
    files = generated_id_files(run_dir)
    results = []
    status = 0
    for path in files:
        result = compare_ids(path, args.vllm_ids, args.prefix_len, args.max_period)
        if args.expect_hash:
            result["expected_hash"] = args.expect_hash
            result["hash_matches"] = (
                result["rtp_hash"] == args.expect_hash
                and result["vllm_hash"] == args.expect_hash
            )
        results.append(result)
        if not result["equal_prefix"]:
            status = max(status, 2)
        if args.expect_hash and not result.get("hash_matches", False):
            status = max(status, 3)
    output = {"run_dir": str(run_dir), "results": results, "ok": status == 0}
    print_json(output, args.json_out)
    return status


def command_run_rtp(args: argparse.Namespace) -> int:
    health_urls = [args.rtp_url]
    if args.prefill_url:
        health_urls.insert(0, args.prefill_url)
    checks = [check_health(url, args.health_timeout) for url in health_urls]
    if not all(item["ok"] for item in checks):
        print_json({"health": checks, "ok": False}, args.json_out)
        return 2

    run_dir = run_rtp_selfroll(args)
    if args.vllm_ids:
        return compare_run_dir(args, run_dir)
    print_json({"run_dir": str(run_dir), "health": checks, "ok": True}, args.json_out)
    return 0


def command_run_teacher(args: argparse.Namespace) -> int:
    args.vllm_ids = args.vllm_ids or DEFAULT_VLLM_ORACLE_IDS
    if args.name is None:
        args.name = (
            f"rtp_teacher_forced_record{args.record_index}_len{args.max_new_tokens}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
    if args.prefix_len is None:
        args.prefix_len = args.max_new_tokens
    print(
        "[teacher] assumes the RTP decode service was restarted with "
        "RTP_TEACHER_FORCE_TOKENS and RTP_TEACHER_FORCE_OFFSET in its process env",
        flush=True,
    )
    return command_run_rtp(args)


def command_known_good(args: argparse.Namespace) -> int:
    if args.compare_only:
        args.rtp_ids = args.rtp_ids or DEFAULT_RTP_FINAL_IDS
        args.vllm_ids = args.vllm_ids or DEFAULT_VLLM_ORACLE_IDS
        args.prefix_len = 1000
        args.expect_hash = args.expect_hash or EXPECTED_RECORD89_HASH1000
        args.fail_on_diff = True
        return command_compare(args)

    args.q_path = Path("/data3/q")
    args.record_index = 89
    args.repeats = args.repeats or 1
    args.max_new_tokens = 1000
    args.stop_repeat_run = 10000
    args.top_k = 1
    args.rtp_mode = "http"
    args.out_root = args.out_root or (DEFAULT_RUN_ROOT / "outputs")
    args.name = args.name or f"rtp_known_good_record89_{time.strftime('%Y%m%d_%H%M%S')}"
    args.vllm_ids = args.vllm_ids or DEFAULT_VLLM_ORACLE_IDS
    args.prefix_len = 1000
    args.expect_hash = args.expect_hash or EXPECTED_RECORD89_HASH1000
    return command_run_rtp(args)


def shell_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def command_write_launch_scripts(args: argparse.Namespace) -> int:
    out_dir = args.output_dir.expanduser().resolve()
    worktree = args.rtp_worktree.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    script_path = Path(__file__).resolve()
    prefill_port = int(args.prefill_port)
    decode_port = int(args.decode_port)
    vllm_port = int(args.vllm_port)
    vllm_python = args.vllm_python
    teacher_force_file: Optional[Path] = None
    teacher_force_info: Optional[Dict[str, Any]] = None
    if args.teacher_force_ids is not None:
        teacher_force_file = (
            args.teacher_force_file.expanduser().resolve()
            if args.teacher_force_file is not None
            else out_dir / "teacher_force_tokens.txt"
        )
        teacher_force_info = write_teacher_force_tokens(
            args.teacher_force_ids.expanduser().resolve(),
            teacher_force_file,
        )

    rtp_pythonpath = f"{worktree}:{worktree / 'bazel-bin'}"
    persistent_topk_env = ""
    if args.dsv4_persistent_topk != "unset":
        persistent_topk_env = f"export DSV4_PERSISTENT_TOPK={int(args.dsv4_persistent_topk)}\n"
    if args.rtp_prefill_topk_backend == "unset":
        prefill_topk_env = "unset DSV4_INDEXER_TOPK_BACKEND\nunset DSV4_TORCH_TOPK"
    else:
        prefill_topk_env = (
            f"export DSV4_INDEXER_TOPK_BACKEND={shell_quote(args.rtp_prefill_topk_backend)}\n"
            "unset DSV4_TORCH_TOPK"
        )
    env_dump_root = args.env_dump_dir.expanduser().resolve() if args.env_dump_dir else None

    common_rtp_env = f"""export PYTHONPATH={shell_quote(rtp_pythonpath)}${{PYTHONPATH:+:${{PYTHONPATH}}}}
BAZEL_BIN=$(readlink -f bazel-bin)
BAZEL_OUTPUT_BASE=${{BAZEL_BIN%%/execroot/rtp_llm/*}}
export LD_LIBRARY_PATH="$BAZEL_OUTPUT_BASE/external/torch_2.11_py310_cuda/torch/lib:/opt/conda310/lib${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
export LOAD_PYTHON_MODEL=1
export LOAD_METHOD=fastsafetensors
export MODEL_TYPE=deepseek_v4
export ACT_TYPE=BF16
export FP8_KV_CACHE=1
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON
export ENABLE_COMM_OVERLAP=0
unset DSV4_TORCH_TOPK
unset DSV4_INDEXER_TOPK_BACKEND
unset DSV4_INDEXER_TOPK_BACKEND_OVERRIDE
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_GATE_FP32={int(args.dsv4_gate_fp32)}
export RTP_STABLE_GREEDY_TIEBREAK=1
export RTP_CUDA_GRAPH_SYNC_REPLAY=1
{persistent_topk_env.rstrip()}
export DSV4_FUSED_PREPARE=0
export DSV4_TORCH_ATTN=0
export USE_LOCAL=1
export REUSE_CACHE=0
export ENABLE_DEVICE_CACHE=1
"""

    check_rtp_ops = f"""/opt/conda310/bin/python - <<'PY'
import pathlib
import torch
import rtp_llm.ops
import libth_transformer
import libth_transformer_config
import librtp_compute_ops
from rtp_llm.ops.compute_ops import rtp_llm_ops
expected = pathlib.Path("bazel-bin").resolve()
for name, module in (
    ("libth_transformer_config", libth_transformer_config),
    ("librtp_compute_ops", librtp_compute_ops),
    ("libth_transformer", libth_transformer),
):
    path = pathlib.Path(module.__file__).resolve()
    print(name, path)
    assert path.parent == expected, (
        f"{{name}} loaded from {{path}}, expected direct child of {{expected}}. "
        "Build //:th_transformer_config, //:rtp_compute_ops, and //:th_transformer "
        "in this worktree/output root before starting RTP."
    )
print("torch", torch.__version__)
assert hasattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32"), (
    "DSV4_GATE_FP32=1 requires cublas_gemm_bf16_bf16_fp32 in librtp_compute_ops. "
    "Rebuild //:rtp_compute_ops with the same output root and ensure PYTHONPATH "
    "loads this worktree's bazel-bin before starting RTP."
)
PY
"""

    model_service_config = (
        '{"service_id":"dsv4-precision","role_endpoints":[{"group":"default",'
        f'"prefill_endpoint":{{"type":"Vipserver","address":"127.0.0.1:{prefill_port}",'
        '"protocol":"http","path":"/"},'
        f'"decode_endpoint":{{"type":"Vipserver","address":"127.0.0.1:{decode_port}",'
        '"protocol":"http","path":"/"}}],"use_local":true}'
    )

    start_vllm = f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={shell_quote(args.vllm_gpu)}
export VLLM_DSV4_TORCH_TOPK=1
export VLLM_DSV4_DISABLE_AUX_STREAMS=1
export VLLM_USE_V1=1
exec {shell_quote(vllm_python)} -m vllm.entrypoints.openai.api_server \\
  --model {shell_quote(model_path)} \\
  --served-model-name {shell_quote(args.vllm_model)} \\
  --host 0.0.0.0 \\
  --port {vllm_port} \\
  --max-model-len {int(args.max_model_len)} \\
  --enforce-eager \\
  --no-enable-prefix-caching
"""

    start_prefill = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shell_quote(worktree)}
export CUDA_VISIBLE_DEVICES={shell_quote(args.prefill_gpus)}
export START_PORT={prefill_port}
export ROLE_TYPE=PREFILL
export ENABLE_CUDA_GRAPH=0
{"export RTP_PRECISION_ENV_DUMP_DIR=" + shell_quote(env_dump_root / "prefill") if env_dump_root else "unset RTP_PRECISION_ENV_DUMP_DIR"}
{common_rtp_env}
{prefill_topk_env}
{check_rtp_ops}
exec /opt/conda310/bin/python -m rtp_llm.start_server \\
  --model_type deepseek_v4 \\
  --checkpoint_path {shell_quote(model_path)} \\
  --tokenizer_path {shell_quote(model_path)} \\
  --load_method fastsafetensors \\
  --max_seq_len {int(args.rtp_max_seq_len)} \\
  --act_type BF16 \\
  --tp_size {int(args.prefill_tp_size)} \\
  --ep_size {int(args.prefill_ep_size)} \\
  --world_size {int(args.prefill_world_size)} \\
  --seq_size_per_block 256 \\
  --role_type PREFILL \\
  --cache_store_rdma_mode 0 \\
  --use_local 1 \\
  --reuse_cache 0 \\
  --enable_device_cache 1 \\
  --fp8_kv_cache 1 \\
  --use_deepep_moe 0 \\
  --use_deepep_low_latency 0 \\
  --cp_rotate_method ALL_GATHER \\
  --concurrency_limit 1 \\
  --load_cache_timeout_ms 900000 \\
  --frontend_server_count 1
"""

    if teacher_force_file is None:
        teacher_env = "unset RTP_TEACHER_FORCE_TOKENS RTP_TEACHER_FORCE_OFFSET"
    else:
        teacher_env = (
            f"export RTP_TEACHER_FORCE_TOKENS={shell_quote(teacher_force_file)}\n"
            f"export RTP_TEACHER_FORCE_OFFSET={int(args.teacher_force_offset)}"
        )
    if args.sampler_logits_dump is None:
        sampler_dump_env = "unset RTP_SAMPLER_LOGITS_DUMP RTP_SAMPLER_LOGITS_DUMP_TOPK"
        sampler_dump_info = None
    else:
        sampler_dump_path = args.sampler_logits_dump.expanduser().resolve()
        sampler_dump_env = (
            f"export RTP_SAMPLER_LOGITS_DUMP={shell_quote(sampler_dump_path)}\n"
            f"export RTP_SAMPLER_LOGITS_DUMP_TOPK={int(args.sampler_logits_dump_topk)}"
        )
        sampler_dump_info = {
            "path": str(sampler_dump_path),
            "topk": int(args.sampler_logits_dump_topk),
        }

    start_decode = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shell_quote(worktree)}
{teacher_env}
{sampler_dump_env}
export CUDA_VISIBLE_DEVICES={shell_quote(args.decode_gpu)}
export START_PORT={decode_port}
export ROLE_TYPE=DECODE
export ENABLE_CUDA_GRAPH=1
export ENABLE_CUDA_GRAPH_OVERRIDE=1
{"export RTP_PRECISION_ENV_DUMP_DIR=" + shell_quote(env_dump_root / "decode") if env_dump_root else "unset RTP_PRECISION_ENV_DUMP_DIR"}
export MODEL_SERVICE_CONFIG={shell_quote(model_service_config)}
{common_rtp_env}
{check_rtp_ops}
exec /opt/conda310/bin/python -m rtp_llm.start_server \\
  --model_type deepseek_v4 \\
  --checkpoint_path {shell_quote(model_path)} \\
  --tokenizer_path {shell_quote(model_path)} \\
  --load_method fastsafetensors \\
  --max_seq_len {int(args.rtp_max_seq_len)} \\
  --enable_cuda_graph 1 \\
  --act_type BF16 \\
  --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 \\
  --seq_size_per_block 256 \\
  --role_type DECODE \\
  --cache_store_rdma_mode 0 \\
  --use_local 1 \\
  --reuse_cache 0 \\
  --enable_device_cache 1 \\
  --fp8_kv_cache 1 \\
  --use_deepep_moe 0 \\
  --use_deepep_low_latency 0 \\
  --cp_rotate_method PREFILL_CP \\
  --concurrency_limit 1 \\
  --load_cache_timeout_ms 900000 \\
  --reserver_runtime_mem_mb 2048 \\
  --frontend_server_count 1
"""

    tmux_start = f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
tmux new-session -d -s {shell_quote(args.tmux_prefix + "_vllm")} "bash $SCRIPT_DIR/start_vllm_stable.sh"
tmux new-session -d -s {shell_quote(args.tmux_prefix + "_prefill")} "bash $SCRIPT_DIR/start_rtp_prefill.sh"
tmux new-session -d -s {shell_quote(args.tmux_prefix + "_decode")} "bash $SCRIPT_DIR/start_rtp_decode.sh"
echo "Started tmux sessions:"
echo "  {args.tmux_prefix}_vllm"
echo "  {args.tmux_prefix}_prefill"
echo "  {args.tmux_prefix}_decode"
echo "Run health check:"
echo "  {script_path} health http://127.0.0.1:{prefill_port} http://127.0.0.1:{decode_port} http://127.0.0.1:{vllm_port}"
"""

    run_gate = f"""#!/usr/bin/env bash
set -euo pipefail
{shell_quote(script_path)} known-good-record89 \\
  --rtp-url http://127.0.0.1:{decode_port} \\
  --prefill-url http://127.0.0.1:{prefill_port} \\
  --out-root {shell_quote(args.out_root)}
"""

    if teacher_force_file is None:
        run_teacher_gate = """#!/usr/bin/env bash
set -euo pipefail
echo "This launch bundle was generated without --teacher-force-ids." >&2
echo "Regenerate with --teacher-force-ids <vllm generated_ids.json> and restart RTP decode." >&2
exit 2
"""
    else:
        run_teacher_gate = f"""#!/usr/bin/env bash
set -euo pipefail
# This gate only verifies teacher-forced localization. It is not a final
# natural self-roll proof.
{shell_quote(script_path)} run-teacher \\
  --rtp-url http://127.0.0.1:{decode_port} \\
  --prefill-url http://127.0.0.1:{prefill_port} \\
  --vllm-ids {shell_quote(args.teacher_force_ids.expanduser().resolve())} \\
  --record-index 89 \\
  --max-new-tokens 1000 \\
  --stop-repeat-run 10000 \\
  --top-k 1 \\
  --prefix-len 1000 \\
  --expect-hash {shell_quote(EXPECTED_RECORD89_HASH1000)} \\
  --out-root {shell_quote(args.out_root)}
"""

    files = {
        "start_vllm_stable.sh": start_vllm,
        "start_rtp_prefill.sh": start_prefill,
        "start_rtp_decode.sh": start_decode,
        "start_all_tmux.sh": tmux_start,
        "run_known_good_gate.sh": run_gate,
        "run_teacher_gate.sh": run_teacher_gate,
    }
    for name, content in files.items():
        write_executable(out_dir / name, content)

    result = {
        "output_dir": str(out_dir),
        "files": {name: str(out_dir / name) for name in files},
        "ports": {
            "prefill": prefill_port,
            "decode": decode_port,
            "vllm": vllm_port,
        },
        "rtp_precision_env": {
            "DSV4_GATE_FP32": int(args.dsv4_gate_fp32),
            "DSV4_INDEXER_TOPK_CANONICALIZE": 1,
            "RTP_PREFILL_DSV4_INDEXER_TOPK_BACKEND": args.rtp_prefill_topk_backend,
            "RTP_DECODE_DSV4_INDEXER_TOPK_BACKEND": "unset",
            "DSV4_PERSISTENT_TOPK": args.dsv4_persistent_topk,
            "RTP_STABLE_GREEDY_TIEBREAK": 1,
            "RTP_CUDA_GRAPH_SYNC_REPLAY": 1,
            "RTP_PRECISION_ENV_DUMP_DIR": str(env_dump_root) if env_dump_root else None,
        },
        "tmux_start": str(out_dir / "start_all_tmux.sh"),
        "known_good_gate": str(out_dir / "run_known_good_gate.sh"),
        "teacher_force": {
            "enabled": teacher_force_file is not None,
            "tokens": str(teacher_force_file) if teacher_force_file is not None else None,
            "offset": int(args.teacher_force_offset),
            "source": teacher_force_info,
        },
        "sampler_logits_dump": sampler_dump_info,
        "note": "Review scripts, then run start_all_tmux.sh; after health is OK, run run_known_good_gate.sh.",
    }
    print_json(result, args.json_out)
    return 0


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rtp-ids", type=Path, required=True)
    parser.add_argument("--vllm-ids", type=Path, required=True)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--max-period", type=int, default=64)
    parser.add_argument("--expect-hash", default=None)
    parser.add_argument("--fail-on-diff", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runner", type=Path, default=None)
    parser.add_argument("--q-path", type=Path, default=Path("/data3/q"))
    parser.add_argument("--record-index", type=int, default=89)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--stop-repeat-run", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--rtp-mode", choices=("http", "grpc"), default="http")
    parser.add_argument("--rtp-url", default="http://127.0.0.1:18880")
    parser.add_argument("--prefill-url", default="http://127.0.0.1:18800")
    parser.add_argument("--grpc-addr", default="127.0.0.1:19408")
    parser.add_argument("--prefill-http-port", type=int, default=None)
    parser.add_argument("--prefill-grpc-port", type=int, default=None)
    parser.add_argument("--decode-http-port", type=int, default=None)
    parser.add_argument("--decode-grpc-port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--health-timeout", type=float, default=2.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "outputs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--vllm-ids", type=Path, default=None)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--max-period", type=int, default=64)
    parser.add_argument("--expect-hash", default=None)
    parser.add_argument("--json-out", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare RTP-vLLM precision gates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare generated_ids.json files")
    add_compare_args(compare_parser)
    compare_parser.set_defaults(func=command_compare)

    run_parser = subparsers.add_parser("run-rtp", help="Run RTP self-roll and optionally compare")
    add_run_args(run_parser)
    run_parser.set_defaults(func=command_run_rtp)

    teacher_run_parser = subparsers.add_parser(
        "run-teacher",
        help=(
            "Run an RTP request against a decode service already started with "
            "RTP_TEACHER_FORCE_* and compare against the vLLM oracle"
        ),
    )
    add_run_args(teacher_run_parser)
    teacher_run_parser.set_defaults(func=command_run_teacher)

    known_parser = subparsers.add_parser(
        "known-good-record89",
        help="Run or verify the known-good 19k record89 DSV4 gate",
    )
    add_run_args(known_parser)
    known_parser.add_argument("--compare-only", action="store_true")
    known_parser.add_argument("--rtp-ids", type=Path, default=None)
    known_parser.set_defaults(func=command_known_good)

    health_parser = subparsers.add_parser("health", help="Check /health endpoints")
    health_parser.add_argument("urls", nargs="+")
    health_parser.add_argument("--timeout", type=float, default=2.0)
    health_parser.add_argument("--json-out", type=Path, default=None)
    health_parser.set_defaults(func=command_health)

    maps_parser = subparsers.add_parser(
        "check-process-libs",
        help="Verify a running RTP process loaded all core .so files from one worktree bazel-bin",
    )
    maps_parser.add_argument("--pid", type=int, required=True)
    maps_parser.add_argument("--rtp-worktree", type=Path, default=Path.cwd())
    maps_parser.add_argument("--json-out", type=Path, default=None)
    maps_parser.set_defaults(func=command_check_process_libs)

    env_parser = subparsers.add_parser(
        "check-env-dump",
        help=(
            "Verify opt-in RTP_PRECISION_ENV_DUMP_DIR JSON files written by "
            "backend/rank processes. This avoids relying on /proc/<pid>/environ."
        ),
    )
    env_parser.add_argument("--dump-dir", type=Path, required=True)
    env_parser.add_argument("--stage", default=None)
    env_parser.add_argument("--expect", action="append", default=[])
    env_parser.add_argument("--require-key", action="append", default=[])
    env_parser.add_argument("--json-out", type=Path, default=None)
    env_parser.set_defaults(func=command_check_env_dump)

    teacher_parser = subparsers.add_parser(
        "teacher-force-file",
        help="Create an RTP_TEACHER_FORCE_TOKENS file from vLLM generated_ids.json",
    )
    teacher_parser.add_argument("--vllm-ids", type=Path, required=True)
    teacher_parser.add_argument("--output", type=Path, required=True)
    teacher_parser.add_argument("--offset", type=int, default=1)
    teacher_parser.add_argument("--json-out", type=Path, default=None)
    teacher_parser.set_defaults(func=command_teacher_force_file)

    scripts_parser = subparsers.add_parser(
        "write-launch-scripts",
        help="Generate stable vLLM/RTP PD launch scripts and a known-good gate script",
    )
    scripts_parser.add_argument("--output-dir", type=Path, required=True)
    scripts_parser.add_argument("--rtp-worktree", type=Path, default=Path.cwd())
    scripts_parser.add_argument("--model-path", type=Path, default=Path("/data3/DeepSeekV4-Flash"))
    scripts_parser.add_argument("--vllm-python", default="/data3/vllm-dsv4-env/bin/python")
    scripts_parser.add_argument("--vllm-model", default="deepseek-v4-flash")
    scripts_parser.add_argument("--vllm-gpu", default="4")
    scripts_parser.add_argument("--prefill-gpus", default="0,3")
    scripts_parser.add_argument("--decode-gpu", default="7")
    scripts_parser.add_argument("--prefill-port", type=int, default=18800)
    scripts_parser.add_argument("--decode-port", type=int, default=18880)
    scripts_parser.add_argument("--vllm-port", type=int, default=18000)
    scripts_parser.add_argument("--max-model-len", type=int, default=32768)
    scripts_parser.add_argument("--rtp-max-seq-len", type=int, default=32768)
    scripts_parser.add_argument("--prefill-tp-size", type=int, default=2)
    scripts_parser.add_argument("--prefill-ep-size", type=int, default=2)
    scripts_parser.add_argument("--prefill-world-size", type=int, default=2)
    scripts_parser.add_argument("--tmux-prefix", default="dsv4_precision")
    scripts_parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "outputs")
    scripts_parser.add_argument("--dsv4-gate-fp32", type=int, choices=(0, 1), default=1)
    scripts_parser.add_argument(
        "--dsv4-persistent-topk",
        choices=("unset", "0", "1"),
        default="0",
        help=(
            "Set DSV4_PERSISTENT_TOPK in RTP services. Precision launch default "
            "is 0; pass unset to preserve the runtime default."
        ),
    )
    scripts_parser.add_argument(
        "--rtp-prefill-topk-backend",
        choices=("torch", "auto", "unset"),
        default="torch",
        help=(
            "TopK backend for RTP PREFILL only. The strict stable baseline uses "
            "torch so prefill TopK selection itself is reference-stable; decode "
            "keeps this switch unset."
        ),
    )
    scripts_parser.add_argument(
        "--env-dump-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory root. When set, RTP backend/rank processes write "
            "their effective precision env JSON under prefill/ and decode/. "
            "Default is off."
        ),
    )
    scripts_parser.add_argument(
        "--teacher-force-ids",
        type=Path,
        default=None,
        help=(
            "Optional vLLM generated_ids.json. When set, create a teacher token "
            "file and export RTP_TEACHER_FORCE_* in start_rtp_decode.sh. "
            "Default is off, preserving natural self-roll behavior."
        ),
    )
    scripts_parser.add_argument("--teacher-force-file", type=Path, default=None)
    scripts_parser.add_argument("--teacher-force-offset", type=int, default=1)
    scripts_parser.add_argument(
        "--sampler-logits-dump",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path. When set, start_rtp_decode.sh exports "
            "RTP_SAMPLER_LOGITS_DUMP so Sampler.cc writes top logits before "
            "teacher forcing. Default is off."
        ),
    )
    scripts_parser.add_argument("--sampler-logits-dump-topk", type=int, default=16)
    scripts_parser.add_argument("--json-out", type=Path, default=None)
    scripts_parser.set_defaults(func=command_write_launch_scripts)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        print(f"[error] runner failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001 - CLI should fail with useful text.
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
