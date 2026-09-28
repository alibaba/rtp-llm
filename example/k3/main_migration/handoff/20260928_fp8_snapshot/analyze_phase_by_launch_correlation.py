#!/usr/bin/env python3
"""Attribute asynchronous GPU work to RTP Prefill phases by CUDA launch ID."""

import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927")
RUNS = {
    "integrated_r29": (
        ROOT / "timeline-64k-integrated-padding-offset-r29-16step-20260928/traces",
        "k3_64k_padding_offset_r29_16steps_wr{rank}_1.json",
        8,
    ),
    "integrated_r32": (
        ROOT / "timeline-64k-integrated-paged-conv-r32-16full-20260928/traces",
        "k3_64k_paged_conv_r32_16full_wr{rank}_2.json",
        8,
    ),
    "feat_55641e09": (
        ROOT / "timeline-64k-feat-3fs-fourlayer-r3/traces",
        "k3_64k_feat_same_input_8steps_wr{rank}_2.json",
        4,
    ),
}
OUT = ROOT / "64k-fourlayer-phase-correlation-audit-20260928.json"


def span_ms(events):
    if not events:
        return 0.0
    return (max(x["ts"] + x["dur"] for x in events) - min(x["ts"] for x in events)) / 1000


def union_ms(events):
    intervals = sorted((x["ts"], x["ts"] + x["dur"]) for x in events)
    if not intervals:
        return 0.0
    total = 0.0
    begin, end = intervals[0]
    for left, right in intervals[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - begin
            begin, end = left, right
    return (total + end - begin) / 1000


def family(name):
    low = name.lower()
    if "nccldevkernel" in low:
        return "NCCL"
    if "attn_res_fp8" in low:
        return "AttnRes.fp8_producer"
    if "attn_res" in low or "attnres" in low:
        return "AttnRes"
    if ("_paged_short_conv_prefill_kernel" in low
            or "_kimi_kda_short_conv_paged_prefill_kernel" in low
            or "_causal_conv1d_fwd_kernel" in low):
        return "KDA.short_conv"
    if "tokenspeed_mlafmha" in low:
        return "MLA.TokenSpeed"
    if "cutlass::fmha::kernel::sm100fmhafwdkernel" in low:
        return "MLA.CUTLASS"
    if "culaopskdasm100delta" in low:
        return "KDA.cuLA_delta"
    if "kda_fwd" in low or "kda_gate" in low or "culaopskdasm100fwd" in low:
        return "KDA.other"
    if "moe_output" in low:
        return "MoE.combine"
    if "mega_moe" in low or "megamoe" in low:
        return "MoE.MegaMoE"
    if "nvjet" in low:
        return "GEMM.NVJet_unattributed"
    if "deep_gemm" in low or "deepgemm" in low:
        return "GEMM.DeepGEMM_unattributed"
    if ("quant" in low or "_rmsnorm_sigmoid_gate" in low
            or "_sigmoid_mul_group128" in low
            or "_rmsnorm_fp8" in low or "_sigmoid_gate_fp8" in low
            or "_kda_output_prefill_fp8" in low):
        return "FP8.producer_or_quant"
    if "direct_copy_kernel" in low:
        return "Tensor.copy"
    if "rtp_llm::fusedcopykernel" in low:
        return "PD.fused_copy"
    return "Other_unattributed"


def summarize(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[family(event["name"])].append(event)
    return {
        "kernel_count": len(events),
        "gpu_span_ms": span_ms(events),
        "gpu_union_ms": union_ms(events),
        "gpu_kernel_sum_ms": sum(x["dur"] for x in events) / 1000,
        "families": {
            name: {"count": len(group), "cumulative_ms": sum(x["dur"] for x in group) / 1000}
            for name, group in grouped.items()
        },
    }


def analyze_one(path, expected_requests):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")) as stream:
        events = json.load(stream)["traceEvents"]
    cpu = [x for x in events if x.get("cat") == "cpu_op"]
    phases = {
        "target": sorted((x for x in cpu if x["name"] == "executor.mtp.prefill_step(target_model_forward)"), key=lambda x: x["ts"]),
        "draft": sorted((x for x in cpu if x["name"] == "executor.mtp.prefill_step(draft_model_forward)"), key=lambda x: x["ts"]),
    }
    if any(len(x) != expected_requests for x in phases.values()):
        raise ValueError(f"unexpected phase counts in {path}: { {k: len(v) for k, v in phases.items()} }")
    parents = [x for x in cpu if x["name"].startswith("executor.mtp.prefill_step(prefill_stream_size=")]
    phases["prefill"] = [
        next(parent for parent in parents if parent["ts"] <= target["ts"] < parent["ts"] + parent["dur"])
        for target in phases["target"]
    ]
    if len({id(x) for x in phases["prefill"]}) != expected_requests:
        raise ValueError(f"Prefill scope does not have a one-to-one target match: {path}")

    launches = defaultdict(list)
    for event in events:
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(event)

    assigned = {phase: [[] for _ in range(expected_requests)] for phase in phases}
    missing_launch = 0
    ambiguous_launch = 0
    for kernel in (x for x in events if x.get("cat") == "kernel"):
        correlation = kernel.get("args", {}).get("correlation")
        matching = launches.get(correlation, [])
        if not matching:
            missing_launch += 1
            continue
        for phase, scopes in phases.items():
            matches = [index for index, scope in enumerate(scopes)
                       if any(scope["ts"] <= launch["ts"] < scope["ts"] + scope["dur"]
                              for launch in matching)]
            if len(matches) == 1:
                assigned[phase][matches[0]].append(kernel)
            elif len(matches) > 1:
                ambiguous_launch += 1
    rows = []
    for index in range(expected_requests):
        row = {phase: summarize(assigned[phase][index]) for phase in phases}
        row["cpu_scope_ms"] = {phase: scopes[index]["dur"] / 1000 for phase, scopes in phases.items()}
        row["gpu_phase_overlap_possible"] = True
        rows.append(row)
    # A known late GPU kernel must be kept with its earlier CPU launch.
    if not any(row["target"]["families"].get("MLA.TokenSpeed", {}).get("count", 0) for row in rows):
        raise ValueError(f"MLA target kernel absent after launch attribution: {path}")
    return {"trace": str(path), "requests": rows,
            "missing_launch_kernel_count": missing_launch,
            "ambiguous_launch_kernel_count": ambiguous_launch}


def main():
    result = {
        "method": "GPU kernel correlation -> CPU CUDA launch -> enclosing CPU target/draft/Prefill scope",
        "timing_basis": "GPU span is first-to-last completion of kernels launched in phase; cumulative family time overlaps across streams",
        "caveats": ["r29, r32 and feat were captured on different machines", "feat uses some FP8 NCCL communication, which violates the final BF16 NCCL contract", "four-layer routing/shapes can differ", "vLLM uses a different annotation contract and is not included here"],
        "runs": {},
    }
    for label, (directory, template, expected_requests) in RUNS.items():
        ranks = [analyze_one(directory / template.format(rank=rank), expected_requests) for rank in range(8)]
        worst_rank = [{phase: max(rank["requests"][index][phase]["gpu_span_ms"] for rank in ranks)
                       for phase in ("prefill", "target", "draft")}
                      for index in range(expected_requests)]
        result["runs"][label] = {
            "ranks": ranks,
            "per_request_max_rank_gpu_span_ms": worst_rank,
            "median_max_rank_gpu_span_ms": {
                phase: statistics.median(row[phase] for row in worst_rank)
                for phase in ("prefill", "target", "draft")
            },
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for label, run in result["runs"].items():
        print(label, {k: round(v, 3) for k, v in run["median_max_rank_gpu_span_ms"].items()},
              "missing_launch", sum(x["missing_launch_kernel_count"] for x in run["ranks"]),
              "ambiguous_launch", sum(x["ambiguous_launch_kernel_count"] for x in run["ranks"]))
    print(OUT)


if __name__ == "__main__":
    main()
