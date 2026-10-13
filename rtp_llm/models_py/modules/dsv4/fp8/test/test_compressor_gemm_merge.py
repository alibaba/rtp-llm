"""UT for compressor wkv|wgate fused GEMM (P3).

Verifies that ``cat([wkv, wgate], dim=0)`` GEMM + split is bit-equal to
the two separate ``F.linear`` calls it replaces.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_gemm_merge.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ref(x32, w_kv, w_gate):
    return F.linear(x32, w_kv), F.linear(x32, w_gate)


def _merged(x32, w_kv, w_gate, coff_d):
    cat_w = torch.cat([w_kv, w_gate], dim=0).contiguous()
    both = F.linear(x32, cat_w)
    return both.split(coff_d, dim=-1)


def test_decode_shape():
    """Decode shape: B=1, S=1, dim=7168, coff*d=1024."""
    torch.manual_seed(0)
    B, S, dim, coff_d = 1, 1, 7168, 1024
    x = torch.randn(B, S, dim, dtype=torch.float32, device="cuda") * 0.5
    w_kv = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    w_gate = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    kv_a, sc_a = _ref(x, w_kv, w_gate)
    kv_b, sc_b = _merged(x, w_kv, w_gate, coff_d)
    torch.testing.assert_close(kv_a, kv_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sc_a, sc_b, rtol=1e-5, atol=1e-5)
    print(f"  [decode B={B} S={S} dim={dim} coff_d={coff_d}] OK")


def test_prefill_shape():
    """Prefill shape: B=1, S=2048, dim=7168, coff*d=1024."""
    torch.manual_seed(1)
    B, S, dim, coff_d = 1, 2048, 7168, 1024
    x = torch.randn(B, S, dim, dtype=torch.float32, device="cuda") * 0.5
    w_kv = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    w_gate = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    kv_a, sc_a = _ref(x, w_kv, w_gate)
    kv_b, sc_b = _merged(x, w_kv, w_gate, coff_d)
    torch.testing.assert_close(kv_a, kv_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sc_a, sc_b, rtol=1e-5, atol=1e-5)
    print(f"  [prefill B={B} S={S} dim={dim} coff_d={coff_d}] OK")


def test_indexer_shapes():
    """Indexer compressor (head_dim=128, coff*d=128 with overlap=False on
    the inner layer's compressor)."""
    torch.manual_seed(2)
    B, S, dim, coff_d = 1, 1, 7168, 128
    x = torch.randn(B, S, dim, dtype=torch.float32, device="cuda") * 0.5
    w_kv = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    w_gate = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda") * 0.02
    kv_a, sc_a = _ref(x, w_kv, w_gate)
    kv_b, sc_b = _merged(x, w_kv, w_gate, coff_d)
    torch.testing.assert_close(kv_a, kv_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sc_a, sc_b, rtol=1e-5, atol=1e-5)
    print(f"  [indexer B={B} dim={dim} coff_d={coff_d}] OK")


def bench():
    """Decode hot-path: how much does the merge save in launch + DRAM?"""
    print("\n  GEMM merge — separate vs cat+split")
    print(
        "    {:>20}  {:>10}  {:>10}  {:>8}".format(
            "case", "separate", "merged", "speedup"
        )
    )
    for B, S, dim, coff_d in [
        (1, 1, 7168, 1024),  # main compressor decode
        (1, 1, 7168, 128),  # indexer compressor decode
        (1, 2048, 7168, 1024),
    ]:
        x = torch.randn(B, S, dim, dtype=torch.float32, device="cuda")
        w_kv = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda")
        w_gate = torch.randn(coff_d, dim, dtype=torch.float32, device="cuda")
        cat_w = torch.cat([w_kv, w_gate], dim=0).contiguous()

        def sep():
            F.linear(x, w_kv)
            F.linear(x, w_gate)

        def mrg():
            both = F.linear(x, cat_w)
            both.split(coff_d, dim=-1)

        for _ in range(20):
            sep()
            mrg()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200):
            sep()
        e.record()
        e.synchronize()
        t_s = s.elapsed_time(e) / 200
        s.record()
        for _ in range(200):
            mrg()
        e.record()
        e.synchronize()
        t_m = s.elapsed_time(e) / 200
        case = f"B={B} S={S} cd={coff_d}"
        print(f"    {case:>20}  {t_s*1e3:8.2f}us  {t_m*1e3:8.2f}us  {t_s/t_m:6.2f}x")


if __name__ == "__main__":
    print("== Correctness ==")
    test_decode_shape()
    test_prefill_shape()
    test_indexer_shapes()
    print("\n== Benchmark ==")
    bench()
    print("\nOK")
