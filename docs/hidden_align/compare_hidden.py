#!/usr/bin/env python3
"""Compare per-layer hidden states between vLLM and RTP-LLM dumps.

Produces a table of per-tensor metrics:
  - cosine similarity (mean across sequence positions)
  - relative L2 error (||v-r||/||v||)
  - max absolute difference
  - correlation coefficient
"""
import json
import os
import sys

import numpy as np
import torch

VLLM_DIR = os.path.join(os.path.dirname(__file__), "vllm_dumps")
RTP_DIR = os.path.join(os.path.dirname(__file__), "rtp_llm_dumps")
OUT_DIR = os.path.dirname(__file__)

PROMPTS = ["short", "medium", "long_4k"]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (
        torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
    ).item()


def cosine_sim_per_row(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
        .mean()
        .item()
    )


def relative_l2(ref: torch.Tensor, test: torch.Tensor) -> float:
    diff = (ref.float() - test.float()).norm()
    ref_norm = ref.float().norm()
    if ref_norm == 0:
        return float("inf") if diff > 0 else 0.0
    return (diff / ref_norm).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def compare_prompt(name: str) -> dict:
    vllm_path = os.path.join(VLLM_DIR, f"{name}.pt")
    rtp_path = os.path.join(RTP_DIR, f"{name}.pt")

    if not os.path.exists(vllm_path):
        print(f"  SKIP {name}: vLLM dump not found")
        return {}
    if not os.path.exists(rtp_path):
        print(f"  SKIP {name}: RTP-LLM dump not found")
        return {}

    v = torch.load(vllm_path, map_location="cpu")
    r = torch.load(rtp_path, map_location="cpu")

    v_tensors = v["tensors"]
    r_tensors = r["tensors"]

    results = {}
    tensor_order = [
        "embed_out",
        "layer00_hidden",
        "layer00_residual",
        "layer00_combined",
        "layer01_hidden",
        "layer01_residual",
        "layer01_combined",
        "layer02_hidden",
        "layer02_residual",
        "layer02_combined",
        "layer03_hidden",
        "layer03_residual",
        "layer03_combined",
        "final_norm",
    ]

    for key in tensor_order:
        if key not in v_tensors or key not in r_tensors:
            continue
        vt = v_tensors[key]
        rt = r_tensors[key]

        if vt.shape != rt.shape:
            results[key] = {"error": f"shape mismatch: {vt.shape} vs {rt.shape}"}
            continue

        results[key] = {
            "shape": list(vt.shape),
            "cosine_sim_global": cosine_sim(vt, rt),
            "cosine_sim_per_row": cosine_sim_per_row(vt, rt),
            "relative_l2": relative_l2(vt, rt),
            "max_abs_diff": max_abs_diff(vt, rt),
            "mean_abs_diff": mean_abs_diff(vt, rt),
            "vllm_abs_max": vt.float().abs().max().item(),
            "rtp_abs_max": rt.float().abs().max().item(),
        }

    # Token comparison
    v_tokens = v.get("output_token_ids", [])
    r_tokens = r.get("output_token_ids", [])
    if isinstance(v_tokens, list) and isinstance(r_tokens, list):
        min_len = min(len(v_tokens), len(r_tokens))
        match_count = sum(
            1 for a, b in zip(v_tokens[:min_len], r_tokens[:min_len]) if a == b
        )
        results["_token_match"] = {
            "vllm_len": len(v_tokens),
            "rtp_len": len(r_tokens),
            "match_count": match_count,
            "match_ratio": match_count / min_len if min_len > 0 else 0,
            "first_diverge": next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(v_tokens[:min_len], r_tokens[:min_len])
                    )
                    if a != b
                ),
                min_len,
            ),
        }

    return results


def print_table(name: str, results: dict):
    print(f"\n{'='*80}")
    print(f"  Prompt: {name}")
    print(f"{'='*80}")
    print(
        f"{'Tensor':<22} {'Shape':<14} {'Cos(row)':<10} {'RelL2':<10} {'MaxDiff':<12} {'MeanDiff':<12}"
    )
    print(f"{'-'*80}")

    for key, metrics in results.items():
        if key == "_token_match":
            continue
        if "error" in metrics:
            print(f"{key:<22} {metrics['error']}")
            continue
        shape_str = "×".join(str(s) for s in metrics["shape"])
        print(
            f"{key:<22} {shape_str:<14} "
            f"{metrics['cosine_sim_per_row']:<10.6f} "
            f"{metrics['relative_l2']:<10.6f} "
            f"{metrics['max_abs_diff']:<12.4f} "
            f"{metrics['mean_abs_diff']:<12.6f}"
        )

    if "_token_match" in results:
        tm = results["_token_match"]
        print(
            f"\n  Token match: {tm['match_count']}/{min(tm['vllm_len'], tm['rtp_len'])} "
            f"({tm['match_ratio']*100:.1f}%), first diverge at position {tm['first_diverge']}"
        )


def main():
    prompts = sys.argv[1:] if len(sys.argv) > 1 else PROMPTS
    all_results = {}

    for name in prompts:
        print(f"\nComparing {name}...")
        results = compare_prompt(name)
        if results:
            all_results[name] = results
            print_table(name, results)

    out_path = os.path.join(OUT_DIR, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
