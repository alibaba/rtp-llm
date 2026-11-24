"""Unit tests and B300 (SM103) single-layer micro-benchmarks for Qwen 4.5-Flash:
  1. SiTU-GLU activation (Triton vs PyTorch FP32 reference)
  2. Gemma RMSNorm (Triton vs PyTorch FP32 reference)
  3. MXFP4 quantize/dequantize roundtrip & Triton SM103 tl.dot_scaled parity
  4. Layer 0 Qwen4_5DenseMLP (H=5120, I=12800)
  5. Layers 1..39 Qwen4_5LatentMoE Triton MXFP4 vs dequant-BF16 ref (small + real shapes)
  6. B300 SM103 real-shape micro-benchmark (Decode M=1,8,32,64; Prefill M=512,2048)
"""

from __future__ import annotations

import gc
import math
import sys
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_5.activation import (
    SituAndMul,
    situ_and_mul_ref,
    situ_and_mul_triton,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.moe import (
    Qwen4_5DenseMLP,
    Qwen4_5LatentMoE,
    gemma_rmsnorm,
    gemma_rmsnorm_ref,
)
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    dequantize_mxfp4,
    dequantize_mxfp4_ref,
    mxfp4_linear,
    mxfp4_moe_forward,
    mxfp4_moe_forward_ref,
    quantize_mxfp4,
)


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    denom = bf.abs().max().item()
    max_rel = max_abs / max(denom, 1e-12)
    cos = F.cosine_similarity(af.reshape(-1), bf.reshape(-1), dim=0).item()
    return max_abs, max_rel, cos


def _bench_cuda_ms(fn, *, warmup: int = 8, iters: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def test_situ_and_mul(device: torch.device) -> None:
    print("\n=== [1] Test SituAndMul (Triton vs PyTorch FP32 ref) ===")
    torch.manual_seed(101)
    act = SituAndMul(gate_beta=4.0, up_beta=25.0, use_triton=True)

    for shape in [(1, 2560), (7, 1280), (16, 12800), (2, 5, 2560), (13, 96)]:
        for dtype in (torch.bfloat16, torch.float32):
            # Standard scale: strict max_abs < 1e-3
            gate_up = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, device=device)
            y_tri = act(gate_up)
            y_ref = situ_and_mul_ref(gate_up, gate_beta=4.0, up_beta=25.0)
            max_abs, max_rel, cos = _diff_stats(y_tri, y_ref)
            assert y_tri.shape == shape and y_tri.dtype == dtype
            assert max_abs < 1e-3, f"cat SituAndMul mismatch ({shape}, {dtype}): {max_abs}"

            g, u = gate_up.chunk(2, dim=-1)
            y_split_tri = situ_and_mul_triton(g, u, gate_beta=4.0, up_beta=25.0)
            max_abs_s, _, _ = _diff_stats(y_split_tri, y_ref)
            assert max_abs_s < 1e-3, f"split SituAndMul mismatch ({shape}, {dtype}): {max_abs_s}"

            # Large dynamic range (*8.0, outputs up to 100): exact <=1 BF16 ULP or FP32 < 1e-4
            gate_up_hi = gate_up * 8.0
            y_hi_tri = act(gate_up_hi)
            y_hi_ref = situ_and_mul_ref(gate_up_hi, gate_beta=4.0, up_beta=25.0)
            if dtype == torch.float32:
                assert (y_hi_tri - y_hi_ref).abs().max().item() < 1e-4
            else:
                ulp_diff = (y_hi_tri.view(torch.uint16).int() - y_hi_ref.view(torch.uint16).int()).abs().max().item()
                assert ulp_diff <= 1, f"BF16 ULP diff={ulp_diff} > 1"

    # Extreme saturation check: |gate| >> 4 -> |g| <= 4; |up| >> 25 -> |u| <= 25
    ext_g = torch.tensor([[100.0, -100.0, 0.0, 4.0]], dtype=torch.float32, device=device)
    ext_u = torch.tensor([[200.0, -200.0, 25.0, 200.0]], dtype=torch.float32, device=device)
    y_ext_tri = situ_and_mul_triton(ext_g, ext_u)
    y_ext_ref = situ_and_mul_ref(ext_g, ext_u)
    assert torch.isfinite(y_ext_tri).all()
    ext_diff = (y_ext_tri - y_ext_ref).abs().max().item()
    assert ext_diff < 1e-5
    # Verify exact analytical value at gate=100, up=200: 4*1*1 * 25*1 = 100.0
    assert abs(y_ext_tri[0, 0].item() - 100.0) < 1e-4

    # Report headline BF16 shape (128, 2560)
    x_big = torch.randn((128, 5120), dtype=torch.bfloat16, device=device) * 5.0
    ma, mr, cos = _diff_stats(situ_and_mul_triton(x_big), situ_and_mul_ref(x_big))
    print(f"PASS SituAndMul (BF16 [128, 5120]->[128, 2560]): max_abs={ma:.3e}, max_rel={mr:.3e}, cos={cos:.8f}")


def test_gemma_rmsnorm(device: torch.device) -> None:
    print("\n=== [2] Test GemmaRMSNorm ((1.0 + weight) in FP32) ===")
    torch.manual_seed(102)
    for M, D in [(1, 2560), (16, 2560), (128, 2560), (7, 128)]:
        x_f32 = torch.randn((M, D), dtype=torch.float32, device=device) * 3.0
        w_f32 = torch.randn((D,), dtype=torch.float32, device=device) * 0.25
        y_f32_tri = gemma_rmsnorm(x_f32, w_f32, eps=1e-6, use_triton=True)
        y_f32_ref = gemma_rmsnorm_ref(x_f32, w_f32, eps=1e-6)
        ma_f32, mr_f32, _ = _diff_stats(y_f32_tri, y_f32_ref)
        assert ma_f32 < 1e-5, f"GemmaRMSNorm FP32 mismatch ({M},{D}): {ma_f32}"

        x = x_f32.to(torch.bfloat16)
        w = w_f32.to(torch.bfloat16)
        y_tri = gemma_rmsnorm(x, w, eps=1e-6, use_triton=True)
        y_ref = gemma_rmsnorm_ref(x, w, eps=1e-6)
        ma, mr, cos = _diff_stats(y_tri, y_ref)
        ulp = (y_tri.view(torch.uint16).int() - y_ref.view(torch.uint16).int()).abs().max().item()
        assert ulp <= 1 and cos > 0.999999, f"GemmaRMSNorm BF16 mismatch ({M},{D}): ma={ma} ulp={ulp}"
    print(
        f"PASS GemmaRMSNorm (M=128, D=2560): FP32 max_abs={ma_f32:.3e} | "
        f"BF16 max_abs={ma:.3e}, max_rel={mr:.3e}, max_ulp={ulp}, cos={cos:.8f}"
    )


def test_mxfp4_format_and_dot_scaled(device: torch.device) -> None:
    print("\n=== [3] Test MXFP4 Quant/Dequant & SM103 tl.dot_scaled Parity ===")
    torch.manual_seed(103)

    # 3a. Exhaustive byte & scale dequantization check (Triton vs PyTorch ref)
    all_bytes = torch.arange(256, dtype=torch.uint8, device=device).repeat(16).reshape(32, 128)
    # Finite FP32 range for all nibbles up to 6.0: scale in [1..252] (6.0 * 2^125 < FLT_MAX)
    scales = torch.randint(1, 253, (32, 8), dtype=torch.uint8, device=device)
    dq_tri = dequantize_mxfp4(all_bytes, scales, dtype=torch.float32)
    dq_ref = dequantize_mxfp4_ref(all_bytes, scales, dtype=torch.float32)
    dq_diff = (dq_tri - dq_ref).abs().max().item()
    assert dq_diff == 0.0, f"Triton vs ref dequant diff={dq_diff}"
    # Overflowing normal scales 253..254 match identically (including +/-inf)
    scales_hi = torch.randint(253, 255, (32, 8), dtype=torch.uint8, device=device)
    assert torch.equal(
        dequantize_mxfp4(all_bytes, scales_hi, dtype=torch.float32),
        dequantize_mxfp4_ref(all_bytes, scales_hi, dtype=torch.float32),
    )

    # Check UE8M0 edge values: scale=0 (2^-127) and scale=255 (NaN)
    edge_b = torch.full((2, 16), 0x24, dtype=torch.uint8, device=device)  # lo=2.0, hi=1.0
    edge_sc = torch.tensor([[0], [255]], dtype=torch.uint8, device=device)
    edge_tri = dequantize_mxfp4(edge_b, edge_sc, dtype=torch.float32)
    edge_ref = dequantize_mxfp4_ref(edge_b, edge_sc, dtype=torch.float32)
    assert (edge_tri[0] - edge_ref[0]).abs().max().item() == 0.0
    assert torch.isnan(edge_tri[1]).all() and torch.isnan(edge_ref[1]).all()

    # 3b. Quantize -> Dequantize exact roundtrip on exact MXFP4 matrices
    w_rand_u8 = torch.randint(0, 256, (128, 1280), dtype=torch.uint8, device=device)
    sc_rand = torch.randint(122, 132, (128, 80), dtype=torch.uint8, device=device)
    w_exact_f32 = dequantize_mxfp4_ref(w_rand_u8, sc_rand, dtype=torch.float32)
    rt_u8, rt_sc = quantize_mxfp4(w_exact_f32)
    rt_f32 = dequantize_mxfp4_ref(rt_u8, rt_sc, dtype=torch.float32)
    rt_diff = (rt_f32 - w_exact_f32).abs().max().item()
    assert rt_diff == 0.0, f"MXFP4 exact roundtrip diff={rt_diff}"

    # 3c. Verify tl.dot_scaled nibble order (low=even K, high=odd K) & UE8M0 scale exactness
    # Exact integer-BF16 activations x exact E2M1 weights -> 0.0 error!
    for M, N, K in [(32, 64, 128), (64, 256, 2560)]:
        a_int = torch.randint(-2, 3, (M, K), device=device).to(torch.bfloat16)
        w_u8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        w_sc = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8, device=device)
        y_tri = mxfp4_linear(a_int, w_u8, w_sc, use_triton=True, out_dtype=torch.float32)
        y_ref = mxfp4_linear(a_int, w_u8, w_sc, use_triton=False, out_dtype=torch.float32)
        exact_diff = (y_tri - y_ref).abs().max().item()
        assert exact_diff == 0.0, f"Exact tl.dot_scaled parity failed: {exact_diff}"

    # 3d. Random Gaussian BF16 activations x random MXFP4 weights (real dim K=2560, N=2560)
    a_gauss = torch.randn((32, 2560), dtype=torch.bfloat16, device=device) * 0.2
    w_u8 = torch.randint(0, 256, (2560, 1280), dtype=torch.uint8, device=device)
    w_sc = torch.randint(120, 126, (2560, 80), dtype=torch.uint8, device=device)
    y_tri_f32 = mxfp4_linear(a_gauss, w_u8, w_sc, use_triton=True, out_dtype=torch.float32)
    y_ref_f32 = mxfp4_linear(a_gauss, w_u8, w_sc, use_triton=False, out_dtype=torch.float32)
    ma, mr, cos = _diff_stats(y_tri_f32, y_ref_f32)
    assert ma < 1e-3, f"tl.dot_scaled Gaussian mismatch: {ma}"
    print(
        f"PASS MXFP4 Format & tl.dot_scaled: dequant_diff={dq_diff:.1e}, "
        f"roundtrip_diff={rt_diff:.1e}, exact_dot_diff=0.0, "
        f"gauss_dot(32x2560x2560) max_abs={ma:.3e} max_rel={mr:.3e} cos={cos:.8f}"
    )


def test_dense_mlp(device: torch.device) -> None:
    print("\n=== [4] Test Layer 0 Qwen4_5DenseMLP (H=5120, I=12800) ===")
    torch.manual_seed(104)
    cfg = Qwen4_5ArchConfig()
    mlp = Qwen4_5DenseMLP.from_config(cfg, is_shared_expert=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.02)

    x = torch.randn((2, 8, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
    y_ref = mlp(x, use_triton=False)
    y_tri = mlp(x, use_triton=True)
    ma1, mr1, cos1 = _diff_stats(y_tri, y_ref)
    assert ma1 < 1e-2 and mr1 < 5e-3 and cos1 > 0.99999, f"DenseMLP Triton vs ref mismatch: {ma1} / {mr1}"

    mlp.prepare_weights(fuse_gate_up=True)
    y_fused = mlp(x, use_triton=True)
    ma2, mr2, cos2 = _diff_stats(y_fused, y_ref)
    assert mr2 < 5e-3 and cos2 > 0.99999, f"DenseMLP fused mismatch: {ma2} / {mr2}"
    print(
        f"PASS Qwen4_5DenseMLP: split max_abs={ma1:.3e} rel={mr1:.3e} | "
        f"fused max_abs={ma2:.3e} rel={mr2:.3e} cos={cos2:.8f}"
    )
    del mlp
    torch.cuda.empty_cache()


def _init_latent_moe_weights(moe: Qwen4_5LatentMoE, seed: int) -> None:
    """Initialize realistic BF16 and MXFP4 weights on GPU (fast, low-memory)."""
    g = torch.Generator(device=moe.latent_down_proj.weight.device)
    g.manual_seed(seed)
    with torch.no_grad():
        for name, p in moe.named_parameters():
            if "expert_bias" in name:
                p.uniform_(-0.2, 0.2, generator=g)
            elif "layernorm" in name:
                p.normal_(0.0, 0.05, generator=g)
            else:
                std = 1.0 / math.sqrt(p.shape[-1]) * 0.5
                p.normal_(0.0, std, generator=g)
        # Realistic MXFP4 expert weights: random nibbles + small UE8M0 scales around 2^-7 .. 2^-5
        moe.w13_u8.random_(0, 256, generator=g)
        moe.w13_scale.random_(120, 123, generator=g)
        moe.w2_u8.random_(0, 256, generator=g)
        moe.w2_scale.random_(120, 123, generator=g)


def test_latent_moe_small(device: torch.device) -> None:
    print("\n=== [5a] Test Qwen4_5LatentMoE Small Shape (E=16, K=4, H=256, L=128) ===")
    moe = Qwen4_5LatentMoE(
        hidden_size=256,
        routed_expert_hidden_size=128,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=64,
        num_experts=16,
        num_experts_per_tok=4,
        dtype=torch.bfloat16,
        device=device,
    )
    _init_latent_moe_weights(moe, seed=201)

    # Verify router semantics & exact fused Triton vs PyTorch parity
    x_route = torch.randn((32, 128), dtype=torch.bfloat16, device=device)
    ids_tri, w_tri = moe.route(x_route, use_triton=True)
    ids_ref, w_ref = moe.route(x_route, use_triton=False)
    assert ids_tri.shape == (32, 4) and w_tri.shape == (32, 4)
    assert w_tri.dtype == torch.float32
    assert torch.equal(ids_tri.to(torch.int64), ids_ref.to(torch.int64)), "Router topk_ids mismatch!"
    assert (w_tri - w_ref).abs().max().item() < 1e-6, f"Router weights diff={(w_tri - w_ref).abs().max().item()}"
    assert (w_tri.sum(dim=-1) - 1.0).abs().max().item() < 1e-2

    # Verify load_mxfp4_weights from separate gate/up/down produces identical weights
    w_gate_u8, w_up_u8 = moe.w13_u8.chunk(2, dim=1)
    w_gate_sc, w_up_sc = moe.w13_scale.chunk(2, dim=1)
    w_down_u8, w_down_sc = moe.w2_u8.clone(), moe.w2_scale.clone()
    moe.load_mxfp4_weights(
        w_gate_u8=w_gate_u8.clone(),
        w_gate_scale=w_gate_sc.clone(),
        w_up_u8=w_up_u8.clone(),
        w_up_scale=w_up_sc.clone(),
        w2_u8=w_down_u8,
        w2_scale=w_down_sc,
    )

    for shape in [(1, 256), (16, 256), (2, 32, 256)]:
        x = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
        y_ref = moe(x, use_triton_mxfp4=False)
        y_tri = moe(x, use_triton_mxfp4=True)
        ma, mr, cos = _diff_stats(y_tri, y_ref)
        assert y_tri.shape == shape and y_tri.dtype == torch.bfloat16
        assert mr < 1e-2 and cos > 0.99999, f"Small LatentMoE mismatch {shape}: ma={ma} mr={mr}"
        print(f"  shape={str(shape):12s} (native [E,N,K/2]): max_abs={ma:.3e}, max_rel={mr:.3e}, cos={cos:.8f}")

    # Also verify transposed layout prepare_weights(transpose_mxfp4=True)
    moe.prepare_weights(transpose_mxfp4=True, fuse_shared_gate_up=True)
    x = torch.randn((32, 256), dtype=torch.bfloat16, device=device) * 0.5
    y_ref = moe(x, use_triton_mxfp4=False)
    y_tri = moe(x, use_triton_mxfp4=True)
    ma, mr, cos = _diff_stats(y_tri, y_ref)
    assert mr < 1e-2 and cos > 0.99999
    print(f"  shape=(32, 256)    (trans  [E,K/2,N]): max_abs={ma:.3e}, max_rel={mr:.3e}, cos={cos:.8f}")
    print("PASS Qwen4_5LatentMoE Small Shape")


def test_and_benchmark_latent_moe_real(device: torch.device) -> None:
    print("\n=== [5b] Test & [6] B300 Micro-Benchmark Qwen4_5LatentMoE Real Shape ===")
    print("Config: H=5120, L=2560, I=2560, E=640, top_k=10, shared=1280, MXFP4 Routed Experts")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    cfg = Qwen4_5ArchConfig()
    moe = Qwen4_5LatentMoE.from_config(cfg, device=device, dtype=torch.bfloat16)
    _init_latent_moe_weights(moe, seed=301)
    moe.prepare_weights(transpose_mxfp4=False, fuse_shared_gate_up=True)

    mem_after_weights = torch.cuda.memory_allocated(device)
    weight_gib = (mem_after_weights - mem_before) / (1024**3)
    mxfp4_gib = moe._mxfp4_weights().nbytes() / (1024**3)
    print(
        f"Single-layer resident memory: total={weight_gib:.3f} GiB "
        f"(MXFP4 routed experts={mxfp4_gib:.3f} GiB, BF16 latent+shared+router={weight_gib - mxfp4_gib:.3f} GiB)"
    )

    # --- Accuracy comparison: Triton BF16xE2M1 vs Dequant-BF16 reference ---
    print("\n--- Accuracy: Triton SM103 BF16xE2M1 vs Dequant-BF16 Reference ---")
    torch.manual_seed(404)
    for M in (1, 16, 128):
        x = torch.randn((M, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
        y_ref = moe(x, use_triton_mxfp4=False)
        y_tri = moe(x, use_triton_mxfp4=True, compute_mode="bf16")
        ma, mr, cos = _diff_stats(y_tri, y_ref)
        assert torch.isfinite(y_tri).all(), f"Non-finite output at M={M}"
        assert mr < 1e-2 and cos > 0.99999, f"Real LatentMoE mismatch M={M}: ma={ma} mr={mr} cos={cos}"
        print(f"  M={M:4d} [BF16xE2M1]: max_abs={ma:.4e}, max_rel={mr:.4e}, cos={cos:.8f}")

        if M == 16:
            y_fp8 = moe(x, use_triton_mxfp4=True, compute_mode="e4m3")
            ma8, mr8, cos8 = _diff_stats(y_fp8, y_ref)
            print(f"  M={M:4d} [E4M3xE2M1]: max_abs={ma8:.4e}, max_rel={mr8:.4e}, cos={cos8:.8f} (W4A8 comparison)")

    del x, y_ref, y_tri
    gc.collect()
    torch.cuda.empty_cache()

    # --- B300 SM103 Micro-Benchmark: Decode (M=1,8,32,64) & Prefill (M=512,2048) ---
    print("\n--- B300 (SM103) Single-Layer Latency & Peak Memory Micro-Benchmark ---")
    hdr = (
        f"{'Mode':8s} {'M':>5s} | {'Full MoE BF16xE2M1':>18s} | {'Routed Only':>12s} | "
        f"{'Routed E4M3xE2M1':>16s} | {'Peak Mem (GiB)':>14s}"
    )
    print(hdr)
    print("-" * len(hdr))

    weights = moe._mxfp4_weights()
    for mode_name, M in [
        ("Decode", 1),
        ("Decode", 8),
        ("Decode", 32),
        ("Decode", 64),
        ("Prefill", 512),
        ("Prefill", 2048),
    ]:
        x = torch.randn((M, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
        routed_in = moe.pre_moe_latent_layernorm(F.linear(x, moe.latent_down_proj.weight))
        routed_in = routed_in * torch.sigmoid(F.linear(x, moe.latent_gate_proj.weight))
        topk_ids, topk_weights = moe.route(routed_in)

        torch.cuda.reset_peak_memory_stats(device)
        iters = 50 if M <= 64 else 20
        ms_full = _bench_cuda_ms(
            lambda: moe(x, use_triton_mxfp4=True, compute_mode="bf16"),
            warmup=6,
            iters=iters,
        )
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        ms_routed_bf16 = _bench_cuda_ms(
            lambda: mxfp4_moe_forward(
                routed_in, topk_ids, topk_weights, weights, compute_mode="bf16", use_triton=True
            ),
            warmup=6,
            iters=iters,
        )
        ms_routed_e4m3 = _bench_cuda_ms(
            lambda: mxfp4_moe_forward(
                routed_in, topk_ids, topk_weights, weights, compute_mode="e4m3", use_triton=True
            ),
            warmup=6,
            iters=iters,
        )
        print(
            f"{mode_name:8s} {M:5d} | {ms_full:15.4f} ms | {ms_routed_bf16:9.4f} ms | "
            f"{ms_routed_e4m3:13.4f} ms | {peak_gib:10.3f} GiB"
        )

    del moe
    gc.collect()
    torch.cuda.empty_cache()

    # Also benchmark Layer 0 Dense MLP (H=5120, I=12800)
    print("\n--- B300 (SM103) Layer 0 Qwen4_5DenseMLP Latency (H=5120, I=12800) ---")
    dense = Qwen4_5DenseMLP.from_config(cfg, is_shared_expert=False, device=device, dtype=torch.bfloat16)
    dense.prepare_weights(fuse_gate_up=True)
    for M in (1, 64, 512, 2048):
        x = torch.randn((M, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
        ms_dense = _bench_cuda_ms(lambda: dense(x), warmup=8, iters=30)
        print(f"  DenseMLP M={M:4d}: {ms_dense:.4f} ms")


def main() -> int:
    assert torch.cuda.is_available(), "CUDA GPU is required"
    device = torch.device("cuda:0")
    print(
        f"Running Qwen 4.5-Flash Latent MoE & MXFP4 tests on "
        f"{torch.cuda.get_device_name(device)} (SM{torch.cuda.get_device_capability(device)})"
    )
    with torch.inference_mode():
        test_situ_and_mul(device)
        test_gemma_rmsnorm(device)
        test_mxfp4_format_and_dot_scaled(device)
        test_dense_mlp(device)
        test_latent_moe_small(device)
        test_and_benchmark_latent_moe_real(device)
    print("\nALL TESTS AND BENCHMARKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
