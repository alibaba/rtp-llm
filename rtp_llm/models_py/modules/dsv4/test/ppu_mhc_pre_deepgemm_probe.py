#!/usr/bin/env python3
"""M890P gate for the projection-only DeepGEMM mHC PRE candidate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import (
    _hc_split_sinkhorn,
    _ppu_deepgemm_linear_mixes,
    _tp_linear_mixes,
)


def _timings_ms(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def _stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "p50_ms": statistics.median(values),
        "mean_ms": mean,
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,16,128,2048,4096")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-replay-abs", type=float, default=1e-5)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--gate-m", type=int, default=4096)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one visible PPU")
    torch.manual_seed(20260828)
    k, n = 16384, 24
    fn = torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.01
    scale = torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float32)
    base = torch.randn((n,), device="cuda", dtype=torch.float32) * 0.01
    module = SimpleNamespace(
        fn=fn,
        norm_eps=1e-6,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        hc_mult=4,
    )
    results = []
    for m in (int(v) for v in args.m_values.split(",")):
        x = (
            torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.2
        ).bfloat16().contiguous()
        reference = _tp_linear_mixes(module, x, use_fp32=True)
        candidate = _ppu_deepgemm_linear_mixes(module, x)
        batched_candidate = _ppu_deepgemm_linear_mixes(module, x.view(1, m, k))
        torch.cuda.synchronize()
        rme = float(
            ((candidate - reference).abs().mean() / reference.abs().mean().clamp_min(1e-12)).item()
        )
        reference_pre, reference_post, reference_comb = _hc_split_sinkhorn(
            reference, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        candidate_pre, candidate_post, candidate_comb = _hc_split_sinkhorn(
            candidate, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        x_hc = x.view(m, 4, k // 4)
        reference_y = torch.sum(
            reference_pre.unsqueeze(-1) * x_hc.float(), dim=-2
        )
        candidate_y = torch.sum(
            candidate_pre.unsqueeze(-1) * x_hc.float(), dim=-2
        )

        def output_rme(got, ref):
            return float(
                ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12)).item()
            )

        pre_output_rme = max(
            output_rme(candidate_y, reference_y),
            output_rme(candidate_post, reference_post),
            output_rme(candidate_comb, reference_comb),
        )

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            side_output = _ppu_deepgemm_linear_mixes(module, x)
        side_stream.synchronize()
        side_max_abs = float((side_output - candidate).abs().max().item())

        _ppu_deepgemm_linear_mixes(module, x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = _ppu_deepgemm_linear_mixes(module, x)
        graph.replay()
        torch.cuda.synchronize()
        graph_max_abs = float((graph_output - candidate).abs().max().item())

        reference_times = _timings_ms(
            lambda: _tp_linear_mixes(module, x, use_fp32=True),
            args.warmup,
            args.iterations,
        )
        candidate_times = _timings_ms(
            lambda: _ppu_deepgemm_linear_mixes(module, x),
            args.warmup,
            args.iterations,
        )
        result = {
            "m": m,
            "rme": rme,
            "pre_output_rme_max": pre_output_rme,
            "finite": bool(torch.isfinite(candidate).all().item()),
            "side_stream_max_abs": side_max_abs,
            "graph_replay_max_abs": graph_max_abs,
            "batched_restore_max_abs": float(
                (batched_candidate.view(m, n) - candidate).abs().max().item()
            ),
            "reference": _stats(reference_times),
            "candidate": _stats(candidate_times),
        }
        result["speedup"] = (
            result["reference"]["p50_ms"] / result["candidate"]["p50_ms"]
        )
        results.append(result)
        del x, reference, candidate, batched_candidate, side_output, graph_output, graph
        del reference_pre, reference_post, reference_comb, reference_y
        del candidate_pre, candidate_post, candidate_comb, candidate_y

    import deep_gemm

    deep_gemm_module = inspect.getsourcefile(deep_gemm)
    if deep_gemm_module is None:
        raise RuntimeError("cannot resolve deep_gemm module path")
    deep_gemm_binary = next(
        path
        for path in Path(deep_gemm_module).parent.glob("deep_gemm_cpp*.so")
    )
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "deep_gemm_module": deep_gemm_module,
        "deep_gemm_binary": str(deep_gemm_binary),
        "deep_gemm_binary_sha256": hashlib.sha256(
            deep_gemm_binary.read_bytes()
        ).hexdigest(),
        "m_values": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if any(
        not row["finite"]
        or row["rme"] >= 1e-3
        or row["pre_output_rme_max"] >= 1e-3
        or row["side_stream_max_abs"] > args.max_replay_abs
        or row["graph_replay_max_abs"] > args.max_replay_abs
        or row["batched_restore_max_abs"] > args.max_replay_abs
        for row in results
    ):
        raise SystemExit(1)
    gate_rows = [row for row in results if row["m"] == args.gate_m]
    if len(gate_rows) != 1:
        raise SystemExit(f"gate M={args.gate_m} must appear exactly once")
    gate = gate_rows[0]
    if gate["candidate"]["cv"] >= 0.02 or gate["speedup"] < 1.0:
        raise SystemExit(
            f"gate M={args.gate_m} failed: candidate_cv={gate['candidate']['cv']}, "
            f"speedup={gate['speedup']}"
        )


if __name__ == "__main__":
    main()
