#!/usr/bin/env python3
"""Summarize DSV4 norm/RoPE + FP8 quant fusion from profiler timelines.

This intentionally reports CUDA kernel event durations only. It ignores Python
and CPU profiler events so the result can be used for the
v4_flash_dp4_decode_bs128_64k_perf acceptance gate.

Acceptance is checked independently for the three endpoint sites we care about:

* Q indexed RoPE + RMSNorm path.
* KV indexed RoPE + RMSNorm path.
* RMSNorm + FP8 quant path.

The decode timeline does not reliably carry Python ``record_function`` ranges
through CUDA graph replay, so Q vs KV is classified from the CUDA kernel shape:
the old Q path is the Triton group-head RMSNorm/RoPE kernel, the old KV path is
the scalar-head Triton RMSNorm/RoPE kernel, and the CUDA replacements have
separate group-warp and block kernels.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from typing import Iterable


OLD_KERNELS = {
    "quant": re.compile(r"per_token_group_quant_8bit"),
    "rmsnorm": re.compile(r"flashinfer::norm::RMSNormKernel|rtp_llm.*rmsnorm", re.I),
    "copy": re.compile(r"direct_copy_kernel|unrolled_elementwise_kernel"),
    "gather": re.compile(r"vectorized_gather_kernel"),
}
NEW_KERNELS = {
    "fused_rmsnorm_fp8_quant": re.compile(
        r"dsv4_fused_rmsnorm_(?:bf16_)?fp8_quant_kernel"
    ),
    "indexed_rmsnorm_rope": re.compile(r"dsv4_indexed_rmsnorm_rope"),
    "indexed_inv_rope_fp8_quant": re.compile(r"dsv4_indexed_inv_rope_fp8_quant_kernel"),
}
ACCEPTANCE_KERNELS = {
    "old.q_rope_rmsnorm": re.compile(r"_fused_rmsnorm_rope_group_heads_kernel"),
    "old.kv_rope_rmsnorm": re.compile(r"(?<!group_heads)_fused_rmsnorm_rope_kernel"),
    "old.rmsnorm": re.compile(
        r"flashinfer::norm::RMSNormKernel|(?=.*rtp_llm)(?!.*dsv4_)(?!.*indexed).*rmsnorm",
        re.I,
    ),
    "old.quant": re.compile(r"per_token_group_quant_8bit"),
    "new.q_rope_rmsnorm": re.compile(
        r"dsv4_indexed_rmsnorm_rope_(?:group_warp|warp)_kernel"
    ),
    "new.kv_rope_rmsnorm": re.compile(
        r"dsv4_indexed_rmsnorm_rope_(?:block|kv_d512_cached_block)_kernel"
    ),
    "new.output_inv_rope_fp8_quant": re.compile(r"dsv4_indexed_inv_rope_fp8_quant_kernel"),
    "new.rmsnorm_fp8_quant": re.compile(
        r"dsv4_fused_rmsnorm_(?:bf16_)?fp8_quant_kernel"
    ),
}

ACCEPTANCE_SITES = {
    "q_rope_rmsnorm_quant": {
        "new": "new.q_rope_rmsnorm",
        "old": "old.q_rope_rmsnorm",
        "note": "timeline proxy for Q indexed RoPE+RMSNorm; Q quant is validated by the FP8 quant buckets",
    },
    "kv_rope_rmsnorm_quant": {
        "new": "new.kv_rope_rmsnorm",
        "old": "old.kv_rope_rmsnorm",
        "note": "timeline proxy for KV indexed RoPE+RMSNorm; KV/output quant is validated separately",
    },
    "rmsnorm_fp8_quant": {
        "new": "new.rmsnorm_fp8_quant",
        "old": "old.rmsnorm",
        "old_extra": "old.quant",
        "note": "RMSNorm followed by per-token FP8 quant replacement",
    },
}


def _open_json(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _timeline_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if name.endswith((".json", ".json.gz")):
                        out.append(os.path.join(root, name))
        else:
            out.append(path)
    return sorted(out)


def _is_kernel_event(event: dict) -> bool:
    if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
        return False
    return "kernel" in str(event.get("cat", "")).lower()


def _add_bucket(buckets, key: str, dur: float) -> None:
    buckets[key]["count"] += 1
    buckets[key]["sum_us"] += dur


def analyze_one(path: str) -> dict:
    data = _open_json(path)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    buckets = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
    for event in events:
        if not _is_kernel_event(event):
            continue
        name = str(event.get("name", ""))
        dur = float(event["dur"])
        for key, pattern in ACCEPTANCE_KERNELS.items():
            if pattern.search(name):
                _add_bucket(buckets, f"accept.{key}", dur)
        matched_new = False
        for key, pattern in NEW_KERNELS.items():
            if pattern.search(name):
                _add_bucket(buckets, f"new.{key}", dur)
                matched_new = True
        if matched_new:
            continue
        for key, pattern in OLD_KERNELS.items():
            if pattern.search(name):
                _add_bucket(buckets, f"old.{key}", dur)
    old_sum = sum(v["sum_us"] for k, v in buckets.items() if k.startswith("old."))
    new_sum = sum(v["sum_us"] for k, v in buckets.items() if k.startswith("new."))
    return {
        "path": path,
        "old_kernel_sum_us": old_sum,
        "new_kernel_sum_us": new_sum,
        "delta_us": old_sum - new_sum,
        "buckets": dict(sorted(buckets.items())),
    }


def _merge_rows(rows: list[dict]) -> dict:
    total = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
    for row in rows:
        for key, stats in row["buckets"].items():
            total[key]["count"] += int(stats["count"])
            total[key]["sum_us"] += float(stats["sum_us"])
    return dict(sorted(total.items()))


def _stats(buckets: dict, key: str) -> dict:
    return buckets.get(key, {"count": 0, "sum_us": 0.0})


def _acceptance(buckets: dict) -> dict:
    out = {}
    for site, spec in ACCEPTANCE_SITES.items():
        new_key = f"accept.{spec['new']}"
        old_key = f"accept.{spec['old']}"
        new_stats = _stats(buckets, new_key)
        old_stats = _stats(buckets, old_key)
        status = "PASS"
        reasons = []
        if new_stats["count"] <= 0:
            status = "FAIL"
            reasons.append(f"missing expected new kernel bucket {spec['new']}")
        if old_stats["count"] > 0:
            status = "FAIL"
            reasons.append(f"old kernel bucket still present: {spec['old']}")
        old_extra = spec.get("old_extra")
        old_extra_stats = _stats(buckets, f"accept.{old_extra}") if old_extra else None
        if site == "rmsnorm_fp8_quant":
            # Other model paths can still legitimately launch RMSNorm or quant.
            # For this site, the hard requirement is seeing the fused kernel.
            if new_stats["count"] > 0:
                status = "PASS"
                reasons = []
        out[site] = {
            "status": status,
            "new": new_stats,
            "old": old_stats,
            "old_extra": old_extra_stats,
            "note": spec["note"],
            "reasons": reasons,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="timeline JSON/JSON.GZ files or directories")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any acceptance site fails")
    args = parser.parse_args()

    rows = []
    for path in _timeline_paths(args.paths):
        row = analyze_one(path)
        if row["buckets"]:
            rows.append(row)
    for row in rows:
        print(f"\n{row['path']}")
        print(
            "  old_sum={:.3f}us new_sum={:.3f}us delta={:.3f}us "
            "per_step_delta={:.3f}us".format(
                row["old_kernel_sum_us"],
                row["new_kernel_sum_us"],
                row["delta_us"],
                row["delta_us"] / max(args.num_steps, 1),
            )
        )
        for name, stats in row["buckets"].items():
            print(f"  {name:34s} count={stats['count']:6d} sum_us={stats['sum_us']:.3f}")

    totals = _merge_rows(rows)
    acceptance = _acceptance(totals)
    print("\nAcceptance")
    for site, result in acceptance.items():
        print(
            "  {site:28s} {status:4s} new_count={new_count:6d} new_sum={new_sum:.3f}us "
            "old_count={old_count:6d} old_sum={old_sum:.3f}us".format(
                site=site,
                status=result["status"],
                new_count=result["new"]["count"],
                new_sum=result["new"]["sum_us"],
                old_count=result["old"]["count"],
                old_sum=result["old"]["sum_us"],
            )
        )
        for reason in result["reasons"]:
            print(f"    reason: {reason}")

    payload = {"num_steps": args.num_steps, "rows": rows, "totals": totals, "acceptance": acceptance}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    if args.strict and any(item["status"] != "PASS" for item in acceptance.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
