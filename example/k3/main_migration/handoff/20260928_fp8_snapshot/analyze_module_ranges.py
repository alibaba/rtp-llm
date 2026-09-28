#!/usr/bin/env python3
"""Attribute all-rank GPU kernels to recorded model scopes by CUDA launch."""

import argparse
import gzip
import io
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

from analyze_phase_by_launch_correlation import family


def union_ms(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    if not intervals:
        return 0.0
    left, right = intervals[0]
    total = 0.0
    for start, stop in intervals[1:]:
        if start <= right:
            right = max(right, stop)
        else:
            total += right - left
            left, right = start, stop
    return (total + right - left) / 1000


def scope_key(scopes):
    if not scopes:
        return "[unattributed]"
    ordered = sorted(scopes, key=lambda e: e["dur"], reverse=True)
    layer = next(
        (re.search(r"RTP::layers\.(\d+)\.", e["name"]).group(1)
         for e in ordered if re.search(r"RTP::layers\.(\d+)\.", e["name"])),
        None,
    )
    label = ordered[-1]["name"]
    return f"L{layer}:{label}" if layer is not None else label


def analyze_trace(events, request_indices, phase):
    phase_name = f"executor.mtp.prefill_step({phase}_model_forward)"
    phases = sorted(
        (e for e in events if e.get("cat") == "cpu_op" and e["name"] == phase_name),
        key=lambda e: e["ts"],
    )
    if not phases or max(request_indices) >= len(phases):
        raise ValueError(f"missing {phase_name} scope or request index")
    annotations = [
        e for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("RTP::")
    ]
    launches = defaultdict(list)
    for event in events:
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(event)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    result = {}
    for index in request_indices:
        parent = phases[index]
        start, stop = parent["ts"], parent["ts"] + parent["dur"]
        annotations_in_phase = [
            e for e in annotations if start <= e["ts"] < stop
        ]
        scoped_by_tid = defaultdict(list)
        for annotation in annotations_in_phase:
            scoped_by_tid[(annotation.get("pid"), annotation.get("tid"))].append(annotation)
        by_module = defaultdict(list)
        ambiguous_launch = 0
        for kernel in kernels:
            correlation = kernel.get("args", {}).get("correlation")
            matching = [
                e for e in launches.get(correlation, ())
                if start <= e["ts"] < stop
            ]
            if not matching:
                continue
            if len(matching) != 1:
                ambiguous_launch += 1
                continue
            launch = matching[0]
            scopes = [
                e for e in scoped_by_tid[(launch.get("pid"), launch.get("tid"))]
                if e["ts"] <= launch["ts"] < e["ts"] + e["dur"]
            ]
            by_module[scope_key(scopes)].append(kernel)
        if not by_module:
            raise ValueError(f"no kernels matched request {index}")
        modules = {}
        for name, selected in sorted(by_module.items()):
            families = defaultdict(float)
            symbols = defaultdict(float)
            for kernel in selected:
                ms = kernel["dur"] / 1000
                families[family(kernel["name"])] += ms
                symbols[kernel["name"]] += ms
            modules[name] = {
                "kernel_count": len(selected),
                "cumulative_ms": sum(e["dur"] for e in selected) / 1000,
                "gpu_union_ms": union_ms(selected),
                "families_cumulative_ms": dict(sorted(families.items())),
                "top_kernel_symbols_ms": sorted(
                    symbols.items(), key=lambda item: item[1], reverse=True
                )[:12],
            }
        result[index] = {
            "modules": modules,
            "labeled_kernel_count": sum(
                value["kernel_count"] for name, value in modules.items()
                if name != "[unattributed]"
            ),
            "unattributed_kernel_count": modules.get("[unattributed]", {}).get(
                "kernel_count", 0
            ),
            "ambiguous_launch": ambiguous_launch,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tar", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("target", "draft"), default="target")
    parser.add_argument("--require-labeled", action="store_true")
    parser.add_argument("--require-scope-fragment", action="append", default=[])
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    matched = audit["matched_requests"]
    if not matched:
        raise ValueError("audit contains no all-rank matched requests")
    rows = [dict() for _ in matched]
    with tarfile.open(args.trace_tar, "r:gz") as archive:
        names = {member.name: member for member in archive if member.isfile()}
        for rank in range(8):
            members = [
                member for name, member in names.items()
                if re.search(rf"_wr{rank}_\d+\.json(?:\.gz)?$", name)
            ]
            if len(members) != 1:
                raise ValueError(f"expected one trace for rank {rank}, got {len(members)}")
            raw = archive.extractfile(members[0]).read()
            if members[0].name.endswith(".gz"):
                raw = gzip.decompress(raw)
            events = json.load(io.BytesIO(raw))["traceEvents"]
            indices = [row["rank_trace_indices"][rank] for row in matched]
            analyzed = analyze_trace(events, set(indices), args.phase)
            for row, index in zip(rows, indices):
                row[str(rank)] = analyzed[index]

    if args.require_labeled and not all(
        rank["labeled_kernel_count"] > 0 for row in rows for rank in row.values()
    ):
        raise ValueError("at least one matched rank/request has no module-labeled kernel")
    for fragment in args.require_scope_fragment:
        if not all(
            any(fragment in name for name in rank["modules"])
            for row in rows for rank in row.values()
        ):
            raise ValueError(f"module scope fragment {fragment!r} is absent on a rank/request")
    keys = sorted({
        module for row in rows for rank in row.values() for module in rank["modules"]
    })
    medians = {}
    for module in keys:
        per_request = [
            max(rank["modules"].get(module, {}).get("cumulative_ms", 0.0)
                for rank in row.values())
            for row in rows
        ]
        medians[module] = statistics.median(per_request)
    output = {
        "phase": args.phase,
        "method": "GPU correlation -> same-thread CUDA launch -> innermost RTP user_annotation; enclosing layer retained",
        "timing_caveat": "cumulative kernel time can overlap across CUDA streams; it is not critical-path latency",
        "trace_tar": str(args.trace_tar),
        "audit": str(args.audit),
        "matched_requests": rows,
        "median_per_request_max_rank_cumulative_ms": dict(
            sorted(medians.items(), key=lambda item: item[1], reverse=True)
        ),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)
    print("matched", len(rows), "labeled", sum(
        rank["labeled_kernel_count"] for row in rows for rank in row.values()
    ))
    for name, ms in list(output["median_per_request_max_rank_cumulative_ms"].items())[:20]:
        print(f"{ms:8.3f} ms {name}")


if __name__ == "__main__":
    main()
