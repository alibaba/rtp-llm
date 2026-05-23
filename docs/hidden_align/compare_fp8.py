#!/usr/bin/env python3
"""3-way comparison: RTP-LLM FP8 vs vLLM FP8 vs vLLM BF16."""
import json
import os

import torch

BASE = os.path.dirname(__file__)
VLLM_BF16_DIR = os.path.join(BASE, "vllm_dumps")
VLLM_FP8_DIR = os.path.join(BASE, "vllm_dumps_fp8")
RTP_FP8_DIR = os.path.join(BASE, "rtp_llm_dumps_fp8")


def cosine_sim_per_row(a, b):
    return (
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
        .mean()
        .item()
    )


def relative_l2(ref, test):
    diff = (ref.float() - test.float()).norm()
    ref_norm = ref.float().norm()
    if ref_norm == 0:
        return float("inf") if diff > 0 else 0.0
    return (diff / ref_norm).item()


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


PROMPTS = ["short", "medium", "long_4k"]
TENSOR_ORDER = [
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


def token_match(a, b):
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0, 0, 0
    match = sum(1 for x, y in zip(a[:min_len], b[:min_len]) if x == y)
    first_div = next(
        (i for i, (x, y) in enumerate(zip(a[:min_len], b[:min_len])) if x != y),
        min_len,
    )
    return match, min_len, first_div


def main():
    results = {}
    for name in PROMPTS:
        vb_path = os.path.join(VLLM_BF16_DIR, f"{name}.pt")
        vf_path = os.path.join(VLLM_FP8_DIR, f"{name}.pt")
        rf_path = os.path.join(RTP_FP8_DIR, f"{name}.pt")

        if not all(os.path.exists(p) for p in [vb_path, vf_path, rf_path]):
            print(f"  SKIP {name}: missing dumps")
            continue

        vllm_bf16 = torch.load(vb_path, map_location="cpu")
        vllm_fp8 = torch.load(vf_path, map_location="cpu")
        rtp_fp8 = torch.load(rf_path, map_location="cpu")

        sep = "=" * 95
        print(f"\n{sep}")
        print(f"  Prompt: {name}")
        print(f"{sep}")
        header = f"{'Tensor':<22} {'RTP-FP8 vs vLLM-FP8':<28} {'RTP-FP8 vs vLLM-BF16':<28} {'vLLM-FP8 vs vLLM-BF16':<28}"
        sub = f"{'':22} {'Cos      RelL2':<28} {'Cos      RelL2':<28} {'Cos      RelL2':<28}"
        print(header)
        print(sub)
        print("-" * 95)

        prompt_results = {}
        for key in TENSOR_ORDER:
            vb = vllm_bf16["tensors"].get(key)
            vf = vllm_fp8["tensors"].get(key)
            rf = rtp_fp8["tensors"].get(key)
            if vb is None or vf is None or rf is None:
                continue

            cos_rv = cosine_sim_per_row(rf, vf)
            rl2_rv = relative_l2(vf, rf)
            cos_rb = cosine_sim_per_row(rf, vb)
            rl2_rb = relative_l2(vb, rf)
            cos_vv = cosine_sim_per_row(vf, vb)
            rl2_vv = relative_l2(vb, vf)

            col1 = f"{cos_rv:.6f}  {rl2_rv:.6f}"
            col2 = f"{cos_rb:.6f}  {rl2_rb:.6f}"
            col3 = f"{cos_vv:.6f}  {rl2_vv:.6f}"
            print(f"{key:<22} {col1:<28} {col2:<28} {col3:<28}")

            prompt_results[key] = {
                "rtp_fp8_vs_vllm_fp8": {"cos": cos_rv, "rel_l2": rl2_rv},
                "rtp_fp8_vs_vllm_bf16": {"cos": cos_rb, "rel_l2": rl2_rb},
                "vllm_fp8_vs_vllm_bf16": {"cos": cos_vv, "rel_l2": rl2_vv},
            }

        # Token comparison
        vb_tok = vllm_bf16.get("output_token_ids", [])
        vf_tok = vllm_fp8.get("output_token_ids", [])
        rf_tok = rtp_fp8.get("output_token_ids", [])

        m_rv, l_rv, d_rv = token_match(rf_tok, vf_tok)
        m_rb, l_rb, d_rb = token_match(rf_tok, vb_tok)
        m_vv, l_vv, d_vv = token_match(vf_tok, vb_tok)
        print(
            f"\n  Tokens: RTP-FP8 vs vLLM-FP8: {m_rv}/{l_rv} ({100*m_rv/max(l_rv,1):.0f}%), first-div@{d_rv}"
        )
        print(
            f"          RTP-FP8 vs vLLM-BF16: {m_rb}/{l_rb} ({100*m_rb/max(l_rb,1):.0f}%), first-div@{d_rb}"
        )
        print(
            f"          vLLM-FP8 vs vLLM-BF16: {m_vv}/{l_vv} ({100*m_vv/max(l_vv,1):.0f}%), first-div@{d_vv}"
        )

        prompt_results["_token_match"] = {
            "rtp_fp8_vs_vllm_fp8": {"match": m_rv, "total": l_rv, "first_div": d_rv},
            "rtp_fp8_vs_vllm_bf16": {"match": m_rb, "total": l_rb, "first_div": d_rb},
            "vllm_fp8_vs_vllm_bf16": {"match": m_vv, "total": l_vv, "first_div": d_vv},
        }
        results[name] = prompt_results

    out_path = os.path.join(BASE, "fp8_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
