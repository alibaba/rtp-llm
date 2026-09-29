#!/usr/bin/env python3
"""Audit archived or live fixed-feat Decode startup after no-CP PD requests.

This is a task-local control-group check. Graph capture and MTP's request
path do not by themselves prove graph replay or GPU draft execution; those
require an all-rank request timeline.
"""

import argparse
import gzip
import json
import math
from pathlib import Path
import re


RANK = re.compile(r"\[RANK (\d+)\]")
CAPTURE = re.compile(r"captured batch[ _]size (\d+)(?:[ :]|$)")
MTP_INPUT = "[mtp-device-input] RTP_LLM_DEVICE_INPUT=1 -> enabled=1"


def physical_graph_buckets(tp: int, proposal_tokens: int) -> list[int]:
    if tp < 1 or proposal_tokens < 1:
        raise ValueError("TP and proposal token count must be positive")
    result = set()
    for width in (1, proposal_tokens + 1):
        alignment = tp // math.gcd(tp, width)
        for size in (1, 2, 4, 8):
            result.add(((size + alignment - 1) // alignment) * alignment)
    return sorted(result)


def audit(env_path: Path, engine_path: Path, tp: int, proposal_tokens: int) -> dict:
    env = {}
    errors = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in env:
            errors.append(f"duplicate service.env key {key}")
        env[key] = value
    expected = {
        "PREFILL_CP_SIZE": "1",
        "DECODE_CP_KV_CACHE_SHARDED": "0",
        "DECODE_CP_Q_REPLICATED": "0",
        "ENABLE_CUDA_GRAPH": "1",
        "FP8_GEMM": "1",
        "FP8_KV_CACHE": "1",
        "FP8_MLA": "1",
        "RTP_LLM_DEVICE_INPUT": "1",
        "SP_TYPE": "mtp",
        "SP_MODEL_TYPE": "kimi_k3_mtp",
        "SP_ACT_TYPE": "BF16",
        "TP_SIZE": str(tp),
        "EP_SIZE": str(tp),
        "GEN_NUM_PER_CIRCLE": str(proposal_tokens),
    }
    for key, value in expected.items():
        if env.get(key) != value:
            errors.append(f"{key}: expected {value}, observed {env.get(key, '<missing>')}")

    buckets = physical_graph_buckets(tp, proposal_tokens)
    captures = {bucket: set() for bucket in buckets}
    mtp_ranks = set()
    reader = gzip.open if engine_path.suffix == ".gz" else open
    with reader(engine_path, "rt", encoding="utf-8", errors="replace") as lines:
        for line in lines:
            match = RANK.search(line)
            if not match:
                continue
            rank = int(match.group(1))
            if "[MLA_DCP]" in line or "[K3_PAGE_RR_TARGET]" in line:
                errors.append(f"rank {rank}: CP/Page-RR marker in no-CP run")
            capture = CAPTURE.search(line)
            if capture:
                bucket = int(capture.group(1))
                if bucket in captures:
                    captures[bucket].add(rank)
            if MTP_INPUT in line:
                mtp_ranks.add(rank)
    required = set(range(tp))
    for bucket in buckets:
        if captures[bucket] != required:
            errors.append(
                f"CUDA Graph bucket {bucket}: missing rank "
                + ",".join(map(str, sorted(required - captures[bucket])))
            )
    if mtp_ranks != required:
        errors.append(
            "MTP request-time device input: missing rank "
            + ",".join(map(str, sorted(required - mtp_ranks)))
        )
    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "graph_buckets": buckets,
        "graph_capture_ranks": {str(k): sorted(v) for k, v in captures.items()},
        "mtp_device_input_ranks": sorted(mtp_ranks),
        "graph_replay_proven": False,
        "mtp_draft_gpu_kernels_proven": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-env", type=Path, required=True)
    parser.add_argument("--engine-log", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--proposal-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.service_env, args.engine_log, args.tp_size, args.proposal_tokens
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"no-CP Decode runtime audit: {report['status']}")
    for error in report["errors"]:
        print(error)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
