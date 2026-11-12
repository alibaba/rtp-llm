"""Standalone HC A/B on synthetic data; never executed by the server.

Run on an otherwise idle SM90 GPU with the matching RTP-LLM environment:
    python -m rtp_llm.models_py.modules.qwen4_exp.test.hc_bf16_down_inject_benchmark
This is a microbenchmark, not model correctness or end-to-end performance.
"""
import json
import statistics

import torch

from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual


def graph_measure(model, x, warmup=10, samples=30):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(x)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return output, times

def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This benchmark requires an available SM90 CUDA device")
    torch.manual_seed(73)
    device = torch.device("cuda")
    width, lowrank = 10240, 320
    gamma = (torch.randn(width, device=device) * 0.01).bfloat16()
    down = (torch.randn(lowrank, width, device=device) * 8).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.002], device=device, dtype=torch.float32)
    up = (torch.randn(width, lowrank, device=device) * 0.005).bfloat16()
    inject = (torch.randn(4, width, device=device) * 0.005).bfloat16()
    baseline = GatedResidual(gamma, down, up, inject, hc_mult=4, norm_eps=1e-6,
        weight_scales=(scale, None, None))
    baseline.bf16_fp8_down_inject = None
    candidate = GatedResidual(gamma, down, up, inject, hc_mult=4, norm_eps=1e-6,
        weight_scales=(scale, None, None))
    if candidate.bf16_fp8_down_inject is None:
        raise RuntimeError("Candidate did not initialize; refusing to benchmark fallback")
    bf16_down = (down.float() * scale).bfloat16()
    reference = GatedResidual(gamma, bf16_down, up, inject, hc_mult=4, norm_eps=1e-6)
    reference.merged_down_inject = None
    print(json.dumps({"candidate_weight_bytes": candidate.bf16_fp8_down_inject.numel() * 2,
        "scope": "one synthetic HC module; includes norm, down/inject, up and mix"}))
    for rows in (12, 2155):
        x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
        candidate_eager = candidate(x)
        reference_eager = reference(x)
        for actual, expected in zip(candidate_eager, reference_eager):
            torch.testing.assert_close(actual, expected, atol=8e-3, rtol=1e-2)
        baseline_eager = baseline(x)
        if not all(bool(torch.isfinite(t).all().item()) for t in candidate_eager + baseline_eager):
            raise RuntimeError("Non-finite output; refusing to benchmark")
        if rows == 12:
            ref, t0 = graph_measure(baseline, x)
            got, t1 = graph_measure(candidate, x)
        else:
            got, t1 = graph_measure(candidate, x)
            ref, t0 = graph_measure(baseline, x)
        diffs = []
        for a, b in zip(got, ref):
            delta = (a.float() - b.float()).abs()
            diffs.append({"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
                "candidate_finite": bool(torch.isfinite(a).all().item()),
                "baseline_finite": bool(torch.isfinite(b).all().item())})
        print(json.dumps({"rows": rows, "baseline_graph_us_p50": statistics.median(t0),
            "candidate_graph_us_p50": statistics.median(t1),
            "candidate_over_baseline": statistics.median(t1) / statistics.median(t0),
            "outputs": diffs}))


if __name__ == "__main__":
    main()
