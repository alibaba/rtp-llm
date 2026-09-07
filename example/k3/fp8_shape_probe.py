"""Probe the installed DeepGEMM with K3 logical shapes; no checkpoint is loaded.

Run each shape in a fresh process so a CUDA failure cannot poison other cases.
This is a correctness probe, not a performance benchmark.
"""

import argparse
import ast
import json
from pathlib import Path

import torch


def load_quantizer(path):
    # Exercise the loader's actual pure tensor function without loading RTP ops.
    tree = ast.parse(Path(path).read_text())
    names = {"ceil_div", "per_block_cast_to_fp8"}
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {"torch": torch, "FP8_E4M3_MAX": 448.0}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["per_block_cast_to_fp8"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantizer",
        default=str(
            Path(__file__).resolve().parents[2]
            / "rtp_llm/model_loader/per_block_fp8_quant_weight.py"
        ),
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    import deep_gemm
    from deep_gemm.utils.layout import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor as pack,
    )

    torch.manual_seed(20260905)
    quantize = load_quantizer(args.quantizer)
    m, n, k = args.m, args.n, args.k
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    wq, ws = quantize(w, 128, use_ue8m0=True)
    grouped = x.float().reshape(m, k // 128, 128)
    xs = grouped.abs().amax(-1).clamp_min(1e-4) / 448.0
    xs = torch.pow(2.0, torch.ceil(torch.log2(xs)))
    xq = (grouped / xs[..., None]).reshape(m, k).to(torch.float8_e4m3fn)
    sf_a = pack(xs)
    expanded_ws = ws.index_select(0, torch.arange(n, device=w.device) // 128)
    sf_b = pack(expanded_ws)
    y = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((xq, sf_a), (wq, sf_b), y)
    torch.cuda.synchronize()
    xd = xq.float() * xs.repeat_interleave(128, -1)
    wd = wq.float() * expanded_ws.repeat_interleave(128, -1)
    reference = xd @ wd.T
    error = (y.float() - reference).norm() / reference.norm().clamp_min(1e-12)
    if not torch.isfinite(y).all() or float(error) > 0.01:
        raise AssertionError(f"invalid FP8 GEMM result: relative L2={float(error)}")
    print(
        json.dumps(
            {
                "m": m,
                "n": n,
                "k": k,
                "weight_shape": list(wq.shape),
                "scale_shape": list(sf_b.shape),
                "output_stride": list(y.stride()),
                "relative_l2": float(error),
                "deep_gemm": deep_gemm.__file__,
                "pass": True,
            }
        )
    )


if __name__ == "__main__":
    main()
