"""Unit tests and B300 (SM103) single-layer micro-benchmarks for Qwen 4.5-Flash:
  1. SiTU-GLU activation (Triton vs PyTorch FP32 reference)
  2. Gemma RMSNorm (Triton vs PyTorch FP32 reference)
  3. MXFP4 quantize/dequantize roundtrip & Triton SM103 tl.dot_scaled parity
  4. Layer 0 Qwen4_5DenseMLP (H=5120, I=12800)
  5. Layers 1..39 Qwen4_5LatentMoE Triton MXFP4 vs dequant-BF16 ref (small + real shapes)
  6. B300 SM103 real-shape micro-benchmark (Decode M=1,8,32,64; Prefill M=512,2048)
  7. Fused SiTU-GLU + Block-32 UE8M0 FP8 Quantization (zero sgl_kernel dep)
  8. SM100 MXFP4 MoE Backends (Triton Split-K, DeepGEMM, FlashInfer) & CUDA Graphs
  9. Qwen4_5LatentMoE aux_stream Shared-Expert Overlap & CUDA Graph Replay
 10. Qwen4_5DenseMLP TP=4 Sharding & Qwen4_5FastAllReduce Distributed Verification
 11. CLI Shootout entrypoint (`--shootout`) for SM100 P-001 empirical benchmarking
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import socket
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_5.activation import (
    SituAndMul,
    per_token_group_quant_fp8_ue8m0,
    per_token_group_quant_fp8_ue8m0_ref,
    situ_and_mul_ref,
    situ_and_mul_triton,
    situ_glu_fp8_quant_masked_packed,
    situ_glu_fp8_quant_packed,
    situ_glu_fp8_quant_ref,
)
from rtp_llm.models_py.modules.qwen4_5.alltoall_comm import (
    Qwen4_5AllToAllComm,
    destroy_qwen4_5_all_to_all,
    get_qwen4_5_all_to_all,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    Qwen4_5FastAllReduce,
    destroy_qwen4_5_fast_all_reduce,
    get_qwen4_5_fast_all_reduce,
)
from rtp_llm.models_py.modules.qwen4_5.fp8_linear import (
    SM100FP8Linear,
    has_deepgemm_fp8,
    has_flashinfer_mxfp8,
    has_flashinfer_trtllm_ll,
    quantize_fp8_rowwise_triton,
    sm100_fp8_linear,
)
from rtp_llm.models_py.modules.qwen4_5.moe import (
    Qwen4_5DenseMLP,
    Qwen4_5LatentMoE,
    gemma_rmsnorm,
    gemma_rmsnorm_ref,
)
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    MXFP4_BLOCK_SIZE,
    MXFP4MoEWeights,
    bench_sm100_shootout,
    dequantize_mxfp4,
    dequantize_mxfp4_ref,
    has_deepgemm_fp8_fp4,
    has_flashinfer_mxfp4,
    mxfp4_linear,
    mxfp4_moe_forward,
    mxfp4_moe_forward_ref,
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_mxfp4_moe_weights,
    quantize_mxfp4,
    resolve_moe_backend_for_shape,
    _resolve_moe_backend,
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

    for shape in [(64, 320), (7, 1280), (1, 2560), (32, 3200), (16, 12800), (2, 5, 2560), (13, 96)]:
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

    # --- Accuracy comparison: Triton BF16xE2M1 + Auto SM100 dispatch vs Dequant-BF16 ref ---
    print("\n--- Accuracy: Triton SM103 BF16xE2M1 & Auto Dispatch vs Dequant-BF16 Reference ---")
    torch.manual_seed(404)
    for M in (1, 16, 128):
        x = torch.randn((M, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
        y_ref = moe(x, use_triton_mxfp4=False)
        y_tri = moe(x, use_triton_mxfp4=True, compute_mode="bf16", backend="triton")
        ma, mr, cos = _diff_stats(y_tri, y_ref)
        assert torch.isfinite(y_tri).all(), f"Non-finite output at M={M}"
        assert mr < 1e-2 and cos > 0.99999, f"Real LatentMoE mismatch M={M}: ma={ma} mr={mr} cos={cos}"
        print(f"  M={M:4d} [BF16xE2M1]: max_abs={ma:.4e}, max_rel={mr:.4e}, cos={cos:.8f}")

        y_auto = moe(x, use_triton_mxfp4=True, compute_mode="bf16", backend="auto")
        ma_a, mr_a, cos_a = _diff_stats(y_auto, y_ref)
        dev_idx = 0 if device.index is None else int(device.index)
        auto_is_fp8 = M > 64 and (has_deepgemm_fp8_fp4(dev_idx) or has_flashinfer_mxfp4(dev_idx))
        if auto_is_fp8:
            # Prefill auto-dispatch selects FP8xMXFP4 W4A8; verify >= 0.9999 vs FP8+UE8M0 ref and >= 0.9985 vs BF16
            r_in = moe.pre_moe_latent_layernorm(F.linear(x, moe.latent_down_proj.weight))
            r_in = r_in * torch.sigmoid(F.linear(x, moe.latent_gate_proj.weight))
            t_ids, t_w = moe.route(r_in)
            r_fp8 = mxfp4_moe_forward_ref(r_in, t_ids, t_w, moe._mxfp4_weights(), gate_beta=moe.gate_beta, up_beta=moe.up_beta, fp8_activations=True)
            r_fp8 = F.linear(r_fp8, moe.latent_up_proj.weight) * torch.sigmoid(F.linear(x, moe.latent_scale_gate.weight))
            s_fp8 = moe._forward_shared(x, run_triton=False)
            y_fp8_ref = r_fp8 + s_fp8
            _, _, cos_fp8 = _diff_stats(y_auto, y_fp8_ref)
            assert cos_fp8 >= 0.9999 and cos_a >= 0.9985, (
                f"Real LatentMoE FP8 auto mismatch M={M}: cos_fp8={cos_fp8}, cos_bf16={cos_a}"
            )
            print(f"  M={M:4d} [auto-SM100]: max_abs={ma_a:.4e}, max_rel={mr_a:.4e}, cos_bf16={cos_a:.8f}, cos_fp8={cos_fp8:.8f}")
        else:
            assert cos_a >= 0.999, f"Real LatentMoE auto-dispatch mismatch M={M}: cos={cos_a}"
            print(f"  M={M:4d} [auto-SM100]: max_abs={ma_a:.4e}, max_rel={mr_a:.4e}, cos={cos_a:.8f}")

        if M == 16:
            y_fp8 = moe(x, use_triton_mxfp4=True, compute_mode="e4m3", backend="triton")
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
                routed_in,
                topk_ids,
                topk_weights,
                weights,
                compute_mode="bf16",
                use_triton=True,
                backend="triton",
            ),
            warmup=6,
            iters=iters,
        )
        ms_routed_e4m3 = _bench_cuda_ms(
            lambda: mxfp4_moe_forward(
                routed_in,
                topk_ids,
                topk_weights,
                weights,
                compute_mode="e4m3",
                use_triton=True,
                backend="triton",
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


def test_situ_glu_fp8_quant(device: torch.device) -> None:
    print("\n=== [7] Test Fused SiTU-GLU + Block-32 UE8M0 FP8 Quantization ===")
    torch.manual_seed(707)

    # 7a. Pure-Triton per_token_group_quant_fp8_ue8m0 (zero sgl_kernel dependency)
    for M, K in [(1, 2560), (7, 2560), (33, 128), (128, 2560)]:
        x = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 2.0
        for fmt in ("raw", "deepgemm", "flashinfer"):
            q_tri, sc_tri = per_token_group_quant_fp8_ue8m0(x, scale_format=fmt)
            q_ref, sc_ref = per_token_group_quant_fp8_ue8m0_ref(x, scale_format=fmt)
            assert torch.equal(q_tri.view(torch.uint8), q_ref.view(torch.uint8)), (
                f"per_token_group_quant fp8 bytes mismatch M={M}, K={K}, fmt={fmt}"
            )
            assert torch.equal(sc_tri, sc_ref), (
                f"per_token_group_quant scale mismatch M={M}, K={K}, fmt={fmt}"
            )

    # 7b. Fused situ_glu_fp8_quant_packed (packed & split, all scale formats + m_indptr)
    for M, D in [(1, 2560), (19, 2560), (128, 2560)]:
        gate_up = torch.randn((M, 2 * D), dtype=torch.bfloat16, device=device) * 2.5
        for fmt in ("raw", "deepgemm", "flashinfer"):
            q_p, s_p = situ_glu_fp8_quant_packed(gate_up, scale_format=fmt)
            q_r, s_r = situ_glu_fp8_quant_ref(gate_up, scale_format=fmt)
            assert torch.equal(q_p.view(torch.uint8), q_r.view(torch.uint8))
            assert torch.equal(s_p, s_r)

        g_t, u_t = gate_up[:, :D].contiguous(), gate_up[:, D:].contiguous()
        q_sp, s_sp = situ_glu_fp8_quant_packed(g_t, u_t, scale_format="deepgemm")
        q_r2, s_r2 = situ_glu_fp8_quant_ref(g_t, u_t, scale_format="deepgemm")
        assert torch.equal(q_sp.view(torch.uint8), q_r2.view(torch.uint8))
        assert torch.equal(s_sp, s_r2)

    # Empty and non-empty groups with FlashInfer m_indptr
    m_indptr = torch.tensor([0, 8, 8, 136, 136], dtype=torch.int32, device=device)
    gu_ind = torch.randn((136, 5120), dtype=torch.bfloat16, device=device) * 1.5
    q_fi, s_fi = situ_glu_fp8_quant_packed(gu_ind, scale_format="flashinfer", m_indptr=m_indptr)
    q_fi_r, s_fi_r = situ_glu_fp8_quant_ref(gu_ind, scale_format="flashinfer", m_indptr=m_indptr)
    assert torch.equal(q_fi.view(torch.uint8), q_fi_r.view(torch.uint8))
    assert torch.equal(s_fi, s_fi_r)

    # 7c. 3D masked situ_glu_fp8_quant_masked_packed & per_token_group_quant_fp8_ue8m0
    #     (persistent active-row dispatch, negative/clamped/zero masked_m, CG replay, >1024 chunked fallback)
    for E_m, max_m_val, D_m, mm_vals, zero_unused in [
        (16, 128, 2560, [0, 3, 12, 0, 128, 1] + [0] * 10, True),
        (160, 512, 2560, [-2, 5, max_m_val := 512, 0] + [0] * 155 + [7], False),
        (1100, 16, 128, [0, 4, 19, 0] + [0] * 1095 + [3], False),
    ]:
        gu_3d = torch.randn((E_m, max_m_val, 2 * D_m), dtype=torch.bfloat16, device=device) * 1.5
        masked_m = torch.tensor(mm_vals, dtype=torch.int32, device=device)
        q_m, s_m = situ_glu_fp8_quant_masked_packed(
            gu_3d, masked_m, zero_unused_scales=zero_unused
        )
        x_3d = torch.randn((E_m, max_m_val, D_m), dtype=torch.bfloat16, device=device) * 1.5
        qx_m, sx_m = per_token_group_quant_fp8_ue8m0(
            x_3d, masked_m=masked_m, scale_format="deepgemm"
        )
        for e in range(E_m):
            cnt = max(0, min(int(masked_m[e].item()), max_m_val))
            if cnt > 0:
                q_e_ref, s_e_ref = situ_glu_fp8_quant_ref(
                    gu_3d[e, :cnt], scale_format="deepgemm"
                )
                assert torch.equal(q_m[e, :cnt].view(torch.uint8), q_e_ref.view(torch.uint8))
                assert torch.equal(s_m[e, :cnt], s_e_ref)
                qx_ref, sx_ref = per_token_group_quant_fp8_ue8m0_ref(
                    x_3d[e, :cnt], scale_format="deepgemm"
                )
                assert torch.equal(qx_m[e, :cnt].view(torch.uint8), qx_ref.view(torch.uint8))
                assert torch.equal(sx_m[e, :cnt], sx_ref)
        del gu_3d, x_3d, q_m, s_m, qx_m, sx_m

    # All-zero masked_m + CUDA Graph capture/replay on E=160 persistent masked quant
    gu_cg = torch.randn((160, 128, 5120), dtype=torch.bfloat16, device=device) * 1.2
    mm_zero = torch.zeros((160,), dtype=torch.int32, device=device)
    _, s_z = situ_glu_fp8_quant_masked_packed(gu_cg, mm_zero, zero_unused_scales=True)
    assert int(s_z.abs().sum().item()) == 0
    mm_cg = torch.zeros((160,), dtype=torch.int32, device=device)
    mm_cg[5], mm_cg[77] = 9, 17
    for _ in range(2):
        situ_glu_fp8_quant_masked_packed(gu_cg, mm_cg, zero_unused_scales=False)
    torch.cuda.synchronize(device)
    g_q = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_q):
        q_cg, s_cg = situ_glu_fp8_quant_masked_packed(gu_cg, mm_cg, zero_unused_scales=False)
    gu_cg2 = torch.randn((160, 128, 5120), dtype=torch.bfloat16, device=device) * 1.2
    gu_cg.copy_(gu_cg2)
    g_q.replay()
    torch.cuda.synchronize(device)
    for e in (5, 77):
        cnt = int(mm_cg[e].item())
        q_r, s_r = situ_glu_fp8_quant_ref(gu_cg2[e, :cnt], scale_format="deepgemm")
        assert torch.equal(q_cg[e, :cnt].view(torch.uint8), q_r.view(torch.uint8))
        assert torch.equal(s_cg[e, :cnt], s_r)
    del g_q, gu_cg, gu_cg2, q_cg, s_cg

    # Verify max_active_rows + zero_unused_scales=False on situ_glu_fp8_quant_packed
    gu_cap = torch.randn((32, 2560), dtype=torch.bfloat16, device=device) * 1.1
    active_rows = torch.tensor([11], dtype=torch.int32, device=device)
    q_cap, s_cap = situ_glu_fp8_quant_packed(
        gu_cap, zero_unused_scales=False, max_active_rows=active_rows
    )
    q_cap_ref, s_cap_ref = situ_glu_fp8_quant_ref(gu_cap[:11], scale_format="deepgemm")
    assert torch.equal(q_cap[:11].view(torch.uint8), q_cap_ref.view(torch.uint8))
    assert torch.equal(s_cap[:11], s_cap_ref)
    del gu_cap, q_cap, s_cap

    print("PASS Fused SiTU-GLU + Block-32 UE8M0 FP8 Quantization")


def test_sm100_mxfp4_backends_and_cuda_graph(device: torch.device) -> None:
    print("\n=== [8-9] Test SM100 MXFP4 Backends, Shared Overlap & CUDA Graph Replay ===")
    torch.manual_seed(808)
    dev_idx = 0 if device.index is None else int(device.index)
    use_dg = has_deepgemm_fp8_fp4(dev_idx)
    use_fi = has_flashinfer_mxfp4(dev_idx)
    print(f"  Optional backends detected: deep_gemm={use_dg}, flashinfer={use_fi}")

    if use_dg:
        import deep_gemm

        for E_s, mn_s, k_s in [(16, 256, 2560), (8, 5120, 2560)]:
            sf_u8 = torch.randint(120, 134, (E_s, mn_s, k_s // MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=device)
            ours = prepare_fp4_weight_scale_for_deepgemm(sf_u8, mn=mn_s, k=k_s, num_groups=E_s)
            sf_f32 = torch.pow(2.0, sf_u8.to(torch.float32) - 127.0)
            ref_dg = deep_gemm.transform_sf_into_required_layout(
                sf_f32, mn=mn_s, k=k_s, recipe=(1, 1, 32), num_groups=E_s, is_sfa=False
            )
            assert torch.equal(ours, ref_dg), f"DeepGEMM scale mismatch E={E_s} mn={mn_s}"

    E, K, N, top_k = 16, 2560, 2560, 4
    w1_fp32 = torch.randn((E, 2 * N, K), device=device, dtype=torch.float32) * 0.02
    w2_fp32 = torch.randn((E, K, N), device=device, dtype=torch.float32) * 0.02
    w1_q, w1_s = quantize_mxfp4(w1_fp32)
    w2_q, w2_s = quantize_mxfp4(w2_fp32)

    backends = ["triton", "triton_splitk"]
    if use_dg:
        backends.extend(["deepgemm_contiguous", "deepgemm_masked"])
    if use_fi:
        backends.extend(["flashinfer", "flashinfer_noswap"])

    for ep_size, ep_start, num_local in [(1, 0, E), (4, 4, 4)]:
        w_obj = prepare_mxfp4_moe_weights(
            w13_u8=w1_q[ep_start:ep_start + num_local],
            w13_scale=w1_s[ep_start:ep_start + num_local],
            w2_u8=w2_q[ep_start:ep_start + num_local],
            w2_scale=w2_s[ep_start:ep_start + num_local],
        )
        m_list = [1, 8, 32, 128] if ep_size == 1 else [1, 32]
        for M in m_list:
            x = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 0.5
            scores = torch.randn((M, E), device=device, dtype=torch.float32)
            topk_w, topk_ids = torch.topk(scores.softmax(dim=-1), k=top_k, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            if ep_size == 4 and M == 1:
                topk_ids[0, 0] = ep_start

            ref_bf16 = mxfp4_moe_forward_ref(
                x, topk_ids, topk_w, w_obj, ep_start=ep_start, ep_size=ep_size
            )
            ref_fp8 = mxfp4_moe_forward_ref(
                x, topk_ids, topk_w, w_obj, ep_start=ep_start, ep_size=ep_size, fp8_activations=True
            )
            for b in backends:
                out = mxfp4_moe_forward(
                    x, topk_ids, topk_w, w_obj, ep_start=ep_start, ep_size=ep_size, backend=b
                )
                _, _, cos_bf16 = _diff_stats(out, ref_bf16)
                _, _, cos_fp8 = _diff_stats(out, ref_fp8)
                target_cos = cos_bf16 if b.startswith("triton") else cos_fp8
                assert target_cos >= 0.9999, (
                    f"Backend parity fail EP={ep_size} M={M} b={b}: bf16={cos_bf16:.6f} fp8={cos_fp8:.6f}"
                )
        if ep_size == 4:
            x_e = torch.randn((4, K), device=device, dtype=torch.bfloat16)
            ids_other = torch.tensor([[0, 1, 2, 3]] * 4, device=device, dtype=torch.int32)
            w_other = torch.full((4, 4), 0.25, device=device, dtype=torch.float32)
            for b in backends:
                out_e = mxfp4_moe_forward(
                    x_e, ids_other, w_other, w_obj, ep_start=ep_start, ep_size=ep_size, backend=b
                )
                assert torch.all(out_e == 0), f"Empty EP slice non-zero on {b}"

    # Verify empirical SM103 E_local-aware regime dispatch & sm100_winner (M>=2) toggle
    w_all_chk = prepare_mxfp4_moe_weights(w13_u8=w1_q, w13_scale=w1_s, w2_u8=w2_q, w2_scale=w2_s)
    w_large_e = MXFP4MoEWeights(
        w13_u8=torch.empty((320, 1, 1), dtype=torch.uint8, device="meta"),
        w13_scale=torch.empty((320, 2 * N, 1), dtype=torch.uint8, device="meta"),
        w2_u8=torch.empty((320, 1, 1), dtype=torch.uint8, device="meta"),
        w2_scale=torch.empty((320, K, 1), dtype=torch.uint8, device="meta"),
    )
    prev_dec = os.environ.get("QWEN4_5_MXFP4_DECODE_BACKEND")
    try:
        os.environ.pop("QWEN4_5_MXFP4_DECODE_BACKEND", None)
        assert _resolve_moe_backend(torch.empty((1, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "triton_splitk"
        assert _resolve_moe_backend(torch.empty((16, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "triton_splitk"
        if use_dg:
            # E_local <= 256 (EP>=4): deepgemm_masked up to M<=2048, deepgemm_contiguous for M>2048
            for m_dg in (72, 512, 1024, 2048):
                assert _resolve_moe_backend(torch.empty((m_dg, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "deepgemm_masked"
            assert _resolve_moe_backend(torch.empty((4096, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "deepgemm_contiguous"
            assert _resolve_moe_backend(torch.empty((16384, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "deepgemm_contiguous"
            # E_local > 256 (EP=1..2): deepgemm_masked <= 1024, flashinfer 1025..16383, contiguous >= 16384
            assert _resolve_moe_backend(torch.empty((1024, K), device=device, dtype=torch.bfloat16), w_large_e, "auto", True) == "deepgemm_masked"
            if use_fi:
                assert _resolve_moe_backend(torch.empty((2048, K), device=device, dtype=torch.bfloat16), w_large_e, "auto", True) == "flashinfer"
            assert _resolve_moe_backend(torch.empty((16384, K), device=device, dtype=torch.bfloat16), w_large_e, "auto", True) == "deepgemm_contiguous"
        prev_wg = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD")
        try:
            os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = "1"
            expected_wg = "deepgemm_masked" if use_dg else "triton"
            assert _resolve_moe_backend(torch.empty((512, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == expected_wg
        finally:
            if prev_wg is None:
                os.environ.pop("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", None)
            else:
                os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = prev_wg
        os.environ["QWEN4_5_MXFP4_DECODE_BACKEND"] = "sm100_winner"
        assert _resolve_moe_backend(torch.empty((1, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "triton_splitk"
        if use_dg:
            assert _resolve_moe_backend(torch.empty((2, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "deepgemm_masked"
            assert _resolve_moe_backend(torch.empty((64, K), device=device, dtype=torch.bfloat16), w_all_chk, "auto", True) == "deepgemm_masked"
    finally:
        if prev_dec is None:
            os.environ.pop("QWEN4_5_MXFP4_DECODE_BACKEND", None)
        else:
            os.environ["QWEN4_5_MXFP4_DECODE_BACKEND"] = prev_dec
    del w_all_chk, w_large_e

    # CUDA Graph replay on mxfp4_moe_forward (Decode M<=64 AND Target Verify/Batch M>64)
    w_all = prepare_mxfp4_moe_weights(w13_u8=w1_q, w13_scale=w1_s, w2_u8=w2_q, w2_scale=w2_s)
    if use_dg:
        w_all.get_deepgemm_pair()
    cg_cases = [
        ("auto", 1),
        ("auto", 8),
        ("auto", 72),
        ("auto", 272),
        ("triton_splitk", 1),
        ("triton_splitk", 8),
        ("triton", 72),
    ]
    if use_dg:
        cg_cases.extend([("deepgemm_masked", 1), ("deepgemm_masked", 8), ("deepgemm_masked", 72)])
    for b, M in cg_cases:
        sx = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 0.5
        sids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
        sw = torch.full((M, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
        for _ in range(3):
            mxfp4_moe_forward(sx, sids, sw, w_all, backend=b)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sout = mxfp4_moe_forward(sx, sids, sw, w_all, backend=b)
        nx = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 0.5
        nids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
        nw = torch.rand((M, top_k), device=device, dtype=torch.float32)
        nw = nw / nw.sum(dim=-1, keepdim=True)
        sx.copy_(nx)
        sids.copy_(nids)
        sw.copy_(nw)
        g.replay()
        torch.cuda.synchronize(device)
        eout = mxfp4_moe_forward(nx, nids, nw, w_all, backend=b)
        _, _, cos_cg = _diff_stats(sout, eout)
        assert cos_cg >= 0.9999, f"MoE CUDA Graph replay mismatch on {b} M={M}: cos={cos_cg}"
        if b == "triton_splitk":
            assert torch.equal(sout, eout), f"triton_splitk M={M} CG vs eager non-exact"
        del g
        print(f"  MoE CUDA Graph replay {b:16s} M={M:3d}: cos={cos_cg:.8f}")

    # Full Qwen4_5LatentMoE with aux_stream Shared Expert overlap + CUDA Graph capture/replay
    moe = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        dtype=torch.bfloat16,
        device=device,
        enable_shared_overlap=True,
        backend="triton_splitk",
    )
    _init_latent_moe_weights(moe, seed=909)
    moe.prepare_weights(prepare_deepgemm=use_dg, prepare_flashinfer=use_fi)

    for M in (1, 16):
        hx = torch.randn((M, 5120), dtype=torch.bfloat16, device=device) * 0.5
        moe.enable_shared_overlap = False
        y_seq = moe(hx)
        moe.enable_shared_overlap = True
        y_ovl = moe(hx)
        _, _, cos_ovl = _diff_stats(y_ovl, y_seq)
        assert cos_ovl > 0.99999, f"Overlap vs sequential mismatch M={M}: cos={cos_ovl}"

    # Capture & replay Qwen4_5LatentMoE across Decode (M=8) and Target Verify (M=72 > 64)
    full_cg_cases = [("triton_splitk", 8), ("auto", 8), ("auto", 72), ("triton", 72)]
    if use_dg:
        full_cg_cases.extend([("deepgemm_masked", 8), ("deepgemm_masked", 72)])
    for b, M in full_cg_cases:
        moe.backend = b
        sx_full = torch.randn((M, 5120), dtype=torch.bfloat16, device=device) * 0.5
        for _ in range(3):
            moe(sx_full)
        torch.cuda.synchronize(device)
        g_full = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_full):
            sout_full = moe(sx_full)
        nx_full = torch.randn((M, 5120), dtype=torch.bfloat16, device=device) * 0.5
        sx_full.copy_(nx_full)
        g_full.replay()
        torch.cuda.synchronize(device)
        eout_full = moe(nx_full)
        _, _, cos_full_cg = _diff_stats(sout_full, eout_full)
        assert cos_full_cg >= 0.9999, f"Full LatentMoE CG mismatch {b} M={M}: cos={cos_full_cg}"
        if b == "triton_splitk":
            eout_full2 = moe(nx_full)
            assert torch.equal(eout_full, eout_full2), "triton_splitk LatentMoE run-to-run non-deterministic"
            assert torch.equal(sout_full, eout_full), "triton_splitk LatentMoE CG vs eager diverged"
        del g_full
        print(f"  LatentMoE + aux_stream CG replay {b:16s} M={M:3d}: cos={cos_full_cg:.8f}")

    # Simulate RTP-LLM CG warmup (RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD=1) -> M=512 (bs=64*8) capture
    moe.backend = "auto"
    sx_512 = torch.randn((512, 5120), dtype=torch.bfloat16, device=device) * 0.5
    prev_wg = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD")
    try:
        os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = "1"
        for _ in range(2):
            moe(sx_512)
    finally:
        if prev_wg is None:
            os.environ.pop("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", None)
        else:
            os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = prev_wg
    torch.cuda.synchronize(device)
    g_512 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_512):
        sout_512 = moe(sx_512)
    nx_512 = torch.randn((512, 5120), dtype=torch.bfloat16, device=device) * 0.5
    sx_512.copy_(nx_512)
    g_512.replay()
    torch.cuda.synchronize(device)
    prev_wg = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD")
    try:
        os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = "1"
        eout_512 = moe(nx_512)
    finally:
        if prev_wg is None:
            os.environ.pop("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", None)
        else:
            os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = prev_wg
    _, _, cos_512 = _diff_stats(sout_512, eout_512)
    assert cos_512 >= 0.9999, f"M=512 CG warmup->capture mismatch: cos={cos_512}"
    del g_512
    print(f"  LatentMoE + aux_stream CG replay {'auto(warmup=1)':16s} M=512: cos={cos_512:.8f}")

    # Verify strict_bf16 / QWEN4_5_MXFP4_STRICT_BF16=1 override and deepgemm_contiguous nosync parity
    x_strict = torch.randn((128, 5120), dtype=torch.bfloat16, device=device) * 0.5
    strict_b = resolve_moe_backend_for_shape(
        128,
        2560,
        device,
        w_all,
        "auto",
        True,
        compute_mode="bf16_strict",
    )
    assert strict_b in ("triton_splitk", "triton"), f"Expected triton strict backend, got {strict_b}"
    y_strict_ref = moe(x_strict, use_triton_mxfp4=False, backend="triton")
    y_strict_mode = moe(x_strict, compute_mode="bf16_strict", backend="auto")
    _, _, cos_strict_mode = _diff_stats(y_strict_mode, y_strict_ref)
    assert cos_strict_mode >= 0.9999, f"bf16_strict M=128 cos={cos_strict_mode}"
    prev_strict = os.environ.get("QWEN4_5_MXFP4_STRICT_BF16")
    try:
        os.environ["QWEN4_5_MXFP4_STRICT_BF16"] = "1"
        assert resolve_moe_backend_for_shape(128, 2560, device, w_all, "auto", True) in ("triton_splitk", "triton")
        y_strict_env = moe(x_strict, compute_mode="bf16", backend="auto")
        _, _, cos_strict_env = _diff_stats(y_strict_env, y_strict_ref)
        assert cos_strict_env >= 0.9999, f"QWEN4_5_MXFP4_STRICT_BF16=1 M=128 cos={cos_strict_env}"
    finally:
        if prev_strict is None:
            os.environ.pop("QWEN4_5_MXFP4_STRICT_BF16", None)
        else:
            os.environ["QWEN4_5_MXFP4_STRICT_BF16"] = prev_strict
    if use_dg and has_flashinfer_mxfp4(dev_idx):
        x_dg = torch.randn((2048, 2560), dtype=torch.bfloat16, device=device) * 0.5
        lg_dg = torch.randn((2048, w_all.num_experts), dtype=torch.float32, device=device)
        t_dg = torch.topk(lg_dg, k=10, dim=-1).indices.to(torch.int32)
        tw_dg = torch.softmax(lg_dg.gather(-1, t_dg.long()), dim=-1).to(torch.float32)
        y_dg1 = mxfp4_moe_forward(x_dg, t_dg, tw_dg, w_all, backend="deepgemm_contiguous")
        y_dg2 = mxfp4_moe_forward(x_dg, t_dg, tw_dg, w_all, backend="deepgemm_contiguous")
        y_fi = mxfp4_moe_forward(x_dg, t_dg, tw_dg, w_all, backend="flashinfer")
        _, _, cos_dg_fi = _diff_stats(y_dg1, y_fi)
        assert _diff_stats(y_dg1, y_dg2)[2] >= 0.999999
        assert cos_dg_fi >= 0.9999, f"deepgemm_contiguous vs flashinfer M=2048 cos={cos_dg_fi}"
        del x_dg, lg_dg, t_dg, tw_dg, y_dg1, y_dg2, y_fi

    del moe, w_all, x_strict
    gc.collect()
    torch.cuda.empty_cache()
    print("PASS SM100 MXFP4 Backends, Shared Overlap & CUDA Graph Replay")


def test_dense_mlp_tp_simulated(device: torch.device) -> None:
    print("\n=== [10a] Test Qwen4_5DenseMLP Simulated TP=4 Sharding (I=12800 -> 3200/rank) ===")
    torch.manual_seed(1010)
    cfg = Qwen4_5ArchConfig()
    full_mlp = Qwen4_5DenseMLP.from_config(cfg, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in full_mlp.parameters():
            p.normal_(0.0, 0.02)
    full_mlp.prepare_weights(fuse_gate_up=True)

    x = torch.randn((16, cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.5
    y_full = full_mlp(x)

    ranks = [
        Qwen4_5DenseMLP.from_config(cfg, tp_size=4, tp_rank=r, device=device, dtype=torch.bfloat16)
        for r in range(4)
    ]
    partial_sum = torch.zeros_like(y_full, dtype=torch.float32)
    for r, shard in enumerate(ranks):
        assert shard.full_intermediate_size == 12800
        assert shard.local_intermediate_size == 3200
        shard.load_tp_weights(
            full_mlp.gate_proj.weight.data,
            full_mlp.up_proj.weight.data,
            full_mlp.down_proj.weight.data,
            already_sharded=False,
        )
        shard.prepare_weights(fuse_gate_up=True)
        partial_sum += shard(x, skip_allreduce=True).float()

    y_tp = partial_sum.to(torch.bfloat16)
    ma, mr, cos = _diff_stats(y_tp, y_full)
    assert mr < 1e-2 and cos > 0.99999, f"Simulated TP=4 mismatch: ma={ma} mr={mr} cos={cos}"
    print(f"PASS Simulated DenseMLP TP=4: max_abs={ma:.3e} rel={mr:.3e} cos={cos:.8f}")
    del full_mlp, ranks
    torch.cuda.empty_cache()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _dist_worker(rank: int, world_size: int, port: int, err_q: mp.Queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
        group = dist.group.WORLD
        with torch.inference_mode():
            _dist_worker_body(rank, world_size, device, group)
        err_q.put((rank, None))
    except Exception as exc:
        err_q.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        destroy_qwen4_5_all_to_all()
        destroy_qwen4_5_fast_all_reduce()
        if dist.is_initialized():
            dist.destroy_process_group()


def _dist_worker_body(rank: int, world_size: int, device: torch.device, group) -> None:
    # 1. Qwen4_5FastAllReduce workspace sharing & CUDA Graph across hidden dims (2560, 5120)
    ar = get_qwen4_5_fast_all_reduce(
        group, device, max_tokens=256, hidden_dims=(5120, 2560)
    )
    ar_shared = get_qwen4_5_fast_all_reduce(
        group, device, max_tokens=256, hidden_dims=(5120,)
    )
    assert ar is ar_shared, "Expected _COMM_CACHE to share workspace across max_hidden=5120"
    for H in (2560, 5120):
        for M in (1, 8, 32):
            x = torch.full((M, H), float(rank + 1), dtype=torch.bfloat16, device=device)
            y = ar(x)
            expected = float(world_size * (world_size + 1) // 2)
            assert torch.allclose(y, torch.full_like(y, expected), atol=1e-2, rtol=1e-2), (
                f"rank={rank} H={H} M={M} AllReduce mismatch: {y[0, 0].item()} != {expected}"
            )

        # CUDA Graph replay on Qwen4_5FastAllReduce (verifying FlashInfer PDL/completion flags)
        sx = torch.full((8, H), float(rank + 1), dtype=torch.bfloat16, device=device)
        for _ in range(3):
            ar(sx)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sy = ar(sx)
        sx.fill_(float((rank + 1) * 2))
        g.replay()
        torch.cuda.synchronize(device)
        expected_cg = float(world_size * (world_size + 1))
        assert torch.allclose(sy, torch.full_like(sy, expected_cg), atol=1e-2, rtol=1e-2)
        del g

    # Test FlashInfer oneshot / twoshot env heuristics & explicit twoshot parity (M > world_size)
    if ar.active_backend == "flashinfer":
        assert ar._resolve_flashinfer_oneshot(world_size) in (None, True)
        prev_os = os.environ.get("QWEN4_5_FAST_AR_ONESHOT")
        prev_th = os.environ.get("QWEN4_5_FAST_AR_TWOSHOT_MIN_TOKENS")
        try:
            os.environ["QWEN4_5_FAST_AR_ONESHOT"] = "auto"
            os.environ["QWEN4_5_FAST_AR_TWOSHOT_MIN_TOKENS"] = "64"
            ar_auto_t = Qwen4_5FastAllReduce(
                group, device, backend="flashinfer", max_tokens=256, hidden_dims=(5120, 2560)
            )
            assert ar_auto_t._resolve_flashinfer_oneshot(16) is None, "expected None below threshold"
            assert ar_auto_t._resolve_flashinfer_oneshot(64) is False, "expected False (twoshot) >= threshold"
            assert ar_auto_t._resolve_flashinfer_oneshot(world_size) is None, "expected None below threshold at world_size"
            ar_two_t = Qwen4_5FastAllReduce(
                group, device, backend="flashinfer", max_tokens=256,
                hidden_dims=(5120, 2560), use_oneshot=False,
            )
            assert ar_two_t._resolve_flashinfer_oneshot(world_size) is True, "expected twoshot clamped to oneshot for M <= world_size"
            assert ar_two_t._resolve_flashinfer_oneshot(world_size + 1) is False, "expected twoshot for M > world_size"
            ar_two_t.destroy()
            x_two = torch.full((8, 5120), float(rank + 1), dtype=torch.bfloat16, device=device)
            ar_two = Qwen4_5FastAllReduce(
                group, device, backend="flashinfer", max_tokens=256,
                hidden_dims=(5120, 2560), use_oneshot=False,
            )
            y_two = ar_two(x_two)
            exp_two = float(world_size * (world_size + 1) // 2)
            assert torch.allclose(y_two, torch.full_like(y_two, exp_two), atol=1e-2, rtol=1e-2)
            ar_auto_t.destroy()
            ar_two.destroy()
        finally:
            if prev_os is None:
                os.environ.pop("QWEN4_5_FAST_AR_ONESHOT", None)
            else:
                os.environ["QWEN4_5_FAST_AR_ONESHOT"] = prev_os
            if prev_th is None:
                os.environ.pop("QWEN4_5_FAST_AR_TWOSHOT_MIN_TOKENS", None)
            else:
                os.environ["QWEN4_5_FAST_AR_TWOSHOT_MIN_TOKENS"] = prev_th

    # 2. Real Distributed DenseMLP TP + CUDA Graph
    torch.manual_seed(2026)
    cfg = Qwen4_5ArchConfig()
    ref_mlp = Qwen4_5DenseMLP.from_config(cfg, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in ref_mlp.parameters():
            p.normal_(0.0, 0.02)
    ref_mlp.prepare_weights(fuse_gate_up=True)

    tp_mlp = Qwen4_5DenseMLP.from_config(
        cfg, tp_size=world_size, tp_rank=rank, tp_group=group, fast_all_reduce=ar,
        device=device, dtype=torch.bfloat16,
    )
    tp_mlp.load_tp_weights(
        ref_mlp.gate_proj.weight.data,
        ref_mlp.up_proj.weight.data,
        ref_mlp.down_proj.weight.data,
        already_sharded=False,
    )
    tp_mlp.prepare_weights(fuse_gate_up=True)

    x_mlp = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.5
    y_ref = ref_mlp(x_mlp)
    for _ in range(3):
        tp_mlp(x_mlp)
    torch.cuda.synchronize(device)
    g_mlp = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_mlp):
        y_tp_cg = tp_mlp(x_mlp)
    g_mlp.replay()
    torch.cuda.synchronize(device)
    _, _, cos_tp = _diff_stats(y_tp_cg, y_ref)
    assert cos_tp > 0.9999, f"rank={rank} Distributed DenseMLP TP CG cos={cos_tp}"

    # 3. Real Distributed Qwen4_5LatentMoE EP + aux_stream overlap + CUDA Graph
    ref_moe = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        dtype=torch.bfloat16,
        device=device,
        enable_shared_overlap=False,
        backend="triton_splitk",
    )
    _init_latent_moe_weights(ref_moe, seed=3030)
    ref_moe.prepare_weights()

    ep_moe = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=group,
        tp_size=world_size,
        tp_rank=rank,
        tp_group=group,
        fast_all_reduce=ar,
        enable_shared_overlap=True,
        backend="triton_splitk",
        dtype=torch.bfloat16,
        device=device,
    )
    ep_moe.load_state_dict(
        {k: v for k, v in ref_moe.state_dict().items() if not k.startswith(("w13_", "w2_"))},
        strict=False,
    )
    lo, hi = ep_moe.ep_start, ep_moe.ep_end
    ep_moe.load_mxfp4_weights(
        w13_u8=ref_moe.w13_u8[lo:hi].clone(),
        w13_scale=ref_moe.w13_scale[lo:hi].clone(),
        w2_u8=ref_moe.w2_u8[lo:hi].clone(),
        w2_scale=ref_moe.w2_scale[lo:hi].clone(),
    )
    ep_moe.prepare_weights()

    x_moe = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.5
    y_moe_ref = ref_moe(x_moe)
    for _ in range(3):
        ep_moe(x_moe)
    torch.cuda.synchronize(device)
    g_moe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_moe):
        y_moe_cg = ep_moe(x_moe)
    g_moe.replay()
    torch.cuda.synchronize(device)
    _, _, cos_ep = _diff_stats(y_moe_cg, y_moe_ref)
    assert cos_ep > 0.9999, f"rank={rank} Distributed LatentMoE EP+overlap CG cos={cos_ep}"

    # 4. FastAllReduce GemmaRMSNorm & mHC combine fusions (+ CG)
    torch.manual_seed(4040 + rank)
    x_g = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.25
    res_g = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.25
    torch.manual_seed(4049)
    w_g = torch.randn((5120,), dtype=torch.bfloat16, device=device) * 0.1
    x_g_sum = x_g.clone()
    dist.all_reduce(x_g_sum, op=dist.ReduceOp.SUM, group=group)
    res_ref = x_g_sum + res_g
    norm_ref = gemma_rmsnorm_ref(res_ref, w_g, eps=1e-6)
    norm_fc, res_fc = ar.all_reduce_residual_gemma_rmsnorm(x_g, res_g, w_g, eps=1e-6)
    _, _, cos_gnorm = _diff_stats(norm_fc, norm_ref)
    assert cos_gnorm > 0.999, f"rank={rank} AR+Res+GemmaRMSNorm cos={cos_gnorm}"
    assert torch.allclose(res_fc, res_ref, atol=2e-2, rtol=2e-2)

    g_gnorm = torch.cuda.CUDAGraph()
    for _ in range(2):
        ar.all_reduce_residual_gemma_rmsnorm(x_g, res_g, w_g, eps=1e-6)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(g_gnorm):
        norm_cg, _ = ar.all_reduce_residual_gemma_rmsnorm(x_g, res_g, w_g, eps=1e-6)
    g_gnorm.replay()
    torch.cuda.synchronize(device)
    _, _, cos_gnorm_cg = _diff_stats(norm_cg, norm_ref)
    assert cos_gnorm_cg > 0.999, f"rank={rank} CG AR+Res+GemmaRMSNorm cos={cos_gnorm_cg}"

    streams_hc = torch.randn((8, 4, 5120), dtype=torch.bfloat16, device=device) * 0.2
    h_post_hc = torch.randn((8, 4), dtype=torch.float32, device=device) * 0.1
    hc_ref = (streams_hc.float() + h_post_hc.unsqueeze(-1) * x_g_sum.unsqueeze(1).float()).to(torch.bfloat16)
    hc_fc = ar.all_reduce_hc_combine(x_g, streams_hc, h_post_hc)
    _, _, cos_hc = _diff_stats(hc_fc, hc_ref)
    assert cos_hc > 0.9995, f"rank={rank} AR+HC combine cos={cos_hc}"
    for _ in range(2):
        ar.all_reduce_hc_combine(x_g, streams_hc, h_post_hc)
    torch.cuda.synchronize(device)
    g_hc = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_hc):
        hc_cg = ar.all_reduce_hc_combine(x_g, streams_hc, h_post_hc)
    g_hc.replay()
    torch.cuda.synchronize(device)
    _, _, cos_hc_cg = _diff_stats(hc_cg, hc_ref)
    assert cos_hc_cg > 0.9995, f"rank={rank} CG AR+HC combine cos={cos_hc_cg}"

    # Verify Qwen4_5FastAllReduce.all_gather (dim=-1 and dim=0) eager & CUDA Graph
    ag_in_last = torch.randn((8, 64), dtype=torch.bfloat16, device=device) * 0.25
    ag_chunks = [torch.empty_like(ag_in_last) for _ in range(world_size)]
    dist.all_gather(ag_chunks, ag_in_last, group=group)
    ag_last_ref = torch.cat(ag_chunks, dim=-1)
    ag_last_fc = ar.all_gather(ag_in_last, dim=-1)
    assert torch.equal(ag_last_fc, ag_last_ref), f"rank={rank} all_gather(dim=-1) eager mismatch"
    ag_dim0_ref = torch.cat(ag_chunks, dim=0)
    ag_dim0_fc = ar.all_gather(ag_in_last, dim=0)
    assert torch.equal(ag_dim0_fc, ag_dim0_ref), f"rank={rank} all_gather(dim=0) eager mismatch"
    for _ in range(2):
        ar.all_gather(ag_in_last, dim=-1)
        ar.all_gather(ag_in_last, dim=0)
    torch.cuda.synchronize(device)
    g_ag = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_ag):
        ag_last_cg = ar.all_gather(ag_in_last, dim=-1)
        ag_dim0_cg = ar.all_gather(ag_in_last, dim=0)
    g_ag.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(ag_last_cg, ag_last_ref), f"rank={rank} all_gather(dim=-1) CG mismatch"
    assert torch.equal(ag_dim0_cg, ag_dim0_ref), f"rank={rank} all_gather(dim=0) CG mismatch"
    del g_hc, g_ag

    # 4b. R3 Chunked FastAllReduce (M > fi_ws_max_tokens) & P2P SymmMem AllGather (bf16/fp32/int32 + multi-chunk + CG)
    prev_ws_cap = os.environ.get("QWEN4_5_FAST_AR_WS_MAX_TOKENS")
    prev_ag_bytes = os.environ.get("QWEN4_5_FAST_AG_SYMM_BYTES")
    prev_ag_min = os.environ.get("QWEN4_5_FAST_AG_SYMM_MIN_BYTES")
    try:
        os.environ["QWEN4_5_FAST_AR_WS_MAX_TOKENS"] = "128"
        os.environ["QWEN4_5_FAST_AG_SYMM_BYTES"] = "32768"
        os.environ["QWEN4_5_FAST_AG_SYMM_MIN_BYTES"] = "0"
        ar_r3 = Qwen4_5FastAllReduce(
            group, device, max_tokens=512, hidden_dims=(5120, 2560)
        )
        assert ar_r3._fi_ws_max_tokens == 128
        for H_chk in (5120, 2560):
            M_chk = 320
            torch.manual_seed(7100 + rank + H_chk)
            x_c = torch.randn((M_chk, H_chk), dtype=torch.bfloat16, device=device) * 0.25
            res_c = torch.randn((M_chk, H_chk), dtype=torch.bfloat16, device=device) * 0.25
            x_c_ref = x_c.clone()
            dist.all_reduce(x_c_ref, op=dist.ReduceOp.SUM, group=group)
            y_c = ar_r3(x_c)
            assert _diff_stats(y_c, x_c_ref)[2] > 0.9999, f"Chunked AR H={H_chk} mismatch"
            torch.manual_seed(7199)
            w_c = torch.randn((H_chk,), dtype=torch.bfloat16, device=device) * 0.1
            norm_c, r_out_c = ar_r3.all_reduce_residual_gemma_rmsnorm(x_c, res_c, w_c, eps=1e-6)
            norm_ref_c = gemma_rmsnorm_ref(x_c_ref + res_c, w_c, eps=1e-6)
            assert _diff_stats(norm_c, norm_ref_c)[2] > 0.999
            assert torch.allclose(r_out_c, x_c_ref + res_c, atol=2e-2, rtol=2e-2)
            for _ in range(2):
                ar_r3(x_c)
            torch.cuda.synchronize(device)
            g_chk = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_chk):
                y_c_cg = ar_r3(x_c)
            g_chk.replay()
            torch.cuda.synchronize(device)
            assert _diff_stats(y_c_cg, x_c_ref)[2] > 0.9999
            del g_chk

        # SymmMem P2P all_gather across bf16, fp32, int32 & multi-chunk (32768B cap -> M=128, D=256 bf16 = 65536B)
        for dt_ag in (torch.bfloat16, torch.float32, torch.int32):
            for shp_ag in ((16, 64), (128, 256)):
                if dt_ag == torch.int32:
                    x_ag = torch.randint(-10000, 10000, shp_ag, dtype=dt_ag, device=device)
                else:
                    x_ag = torch.randn(shp_ag, dtype=dt_ag, device=device) * 0.5
                chunks_ag = [torch.empty_like(x_ag) for _ in range(world_size)]
                dist.all_gather(chunks_ag, x_ag, group=group)
                ref_last = torch.cat(chunks_ag, dim=-1)
                ref_d0 = torch.cat(chunks_ag, dim=0)
                out_last = ar_r3.all_gather(x_ag, dim=-1)
                out_d0 = ar_r3.all_gather(x_ag, dim=0)
                assert torch.equal(out_last, ref_last), f"P2P all_gather(dim=-1) dt={dt_ag} shp={shp_ag}"
                assert torch.equal(out_d0, ref_d0), f"P2P all_gather(dim=0) dt={dt_ag} shp={shp_ag}"
                if shp_ag == (16, 64) and dt_ag in (torch.bfloat16, torch.int32):
                    for _ in range(2):
                        ar_r3.all_gather(x_ag, dim=-1)
                    torch.cuda.synchronize(device)
                    g_p2p = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_p2p):
                        out_last_cg = ar_r3.all_gather(x_ag, dim=-1)
                    g_p2p.replay()
                    torch.cuda.synchronize(device)
                    assert torch.equal(out_last_cg, ref_last), f"CG P2P all_gather dt={dt_ag}"
                    del g_p2p
        ar_r3.destroy()
    finally:
        if prev_ws_cap is None:
            os.environ.pop("QWEN4_5_FAST_AR_WS_MAX_TOKENS", None)
        else:
            os.environ["QWEN4_5_FAST_AR_WS_MAX_TOKENS"] = prev_ws_cap
        if prev_ag_bytes is None:
            os.environ.pop("QWEN4_5_FAST_AG_SYMM_BYTES", None)
        else:
            os.environ["QWEN4_5_FAST_AG_SYMM_BYTES"] = prev_ag_bytes
        if prev_ag_min is None:
            os.environ.pop("QWEN4_5_FAST_AG_SYMM_MIN_BYTES", None)
        else:
            os.environ["QWEN4_5_FAST_AG_SYMM_MIN_BYTES"] = prev_ag_min

    # 5. Qwen4_5AllToAllComm: uneven ragged routed_a2a + dense_gather FP8 + symm_mem CG + DP-EP MoE
    a2a = get_qwen4_5_all_to_all(group, device=device, max_tokens_per_rank=64, hidden_dim=2560, top_k=4)
    torch.manual_seed(5050 + rank)
    m_ragged = 13 if rank == 0 else 5
    hx = torch.randn((m_ragged, 2560), dtype=torch.bfloat16, device=device) * 0.25
    hids = torch.randint(0, 16, (m_ragged, 4), dtype=torch.int32, device=device)
    hw = torch.rand((m_ragged, 4), dtype=torch.float32, device=device)
    st_r = a2a.dispatch(hx, hids, hw, num_experts=16, equal_tokens=False)
    assert st_r.mode == "routed_a2a"
    local_lo = rank * 8
    local_hi = local_lo + 8
    in_local = (st_r.topk_ids >= local_lo) & (st_r.topk_ids < local_hi)
    ep_partial = torch.where(in_local.any(dim=-1, keepdim=True), st_r.hidden * 2.0, torch.zeros_like(st_r.hidden))
    comb_r = a2a.combine(ep_partial, st_r)
    exp_scale = in_local_orig = ((hids >= 0) & (hids < 8)) | ((hids >= 8) & (hids < 16))
    # Count how many distinct EP ranks matched each original token
    rank_owners = torch.zeros((m_ragged,), dtype=torch.float32, device=device)
    for r_idx in range(world_size):
        has_r = ((hids >= r_idx * 8) & (hids < (r_idx + 1) * 8)).any(dim=-1)
        rank_owners += has_r.float()
    expected_r = (hx.float() * 2.0 * rank_owners[:, None]).to(torch.bfloat16)
    _, _, cos_a2a_r = _diff_stats(comb_r, expected_r)
    assert cos_a2a_r > 0.9999, f"rank={rank} ragged routed_a2a cos={cos_a2a_r}"

    # Zero-token asymmetric rank (rank 0 has 7, rank 1 has 0)
    m_zero = 7 if rank == 0 else 0
    st_z = a2a.dispatch(
        hx[:m_zero].contiguous(),
        hids[:m_zero].contiguous(),
        hw[:m_zero].contiguous(),
        num_experts=16,
        equal_tokens=None,
    )
    in_loc_z = (st_z.topk_ids >= local_lo) & (st_z.topk_ids < local_hi)
    comb_z = a2a.combine(
        torch.where(in_loc_z.any(dim=-1, keepdim=True), st_z.hidden, torch.zeros_like(st_z.hidden)),
        st_z,
    )
    assert comb_z.shape == (m_zero, 2560)

    # DP-EP Qwen4_5LatentMoE end-to-end inside CUDA Graph (both triton_splitk & FP8-prequant deepgemm_masked)
    dp_backend = (
        "deepgemm_masked"
        if (has_deepgemm_fp8_fp4(device.index or 0) and not ep_moe.weights_transposed)
        else "triton_splitk"
    )
    dp_moe = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=group,
        tp_size=1,
        tp_rank=0,
        all_to_all_comm=a2a,
        dp_equal_tokens=True,
        use_symm_p2p_combine=(a2a.active_backend == "symm_mem"),
        enable_shared_overlap=True,
        backend=dp_backend,
        dtype=torch.bfloat16,
        device=device,
    )
    dp_moe.load_state_dict(ep_moe.state_dict(), strict=True)
    dp_moe.prepare_weights(prepare_deepgemm=(dp_backend == "deepgemm_masked"))

    torch.manual_seed(6060 + rank)
    x_dp = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.4
    y_dp_ref = ref_moe(x_dp)
    for _ in range(3):
        dp_moe(x_dp)
    torch.cuda.synchronize(device)
    g_dp = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_dp):
        y_dp_cg = dp_moe(x_dp)
    g_dp.replay()
    torch.cuda.synchronize(device)
    _, _, cos_dp_ep = _diff_stats(y_dp_cg, y_dp_ref)
    assert cos_dp_ep > 0.996, f"rank={rank} DP-EP LatentMoE ({dp_backend}) CG cos={cos_dp_ep}"

    # Deterministic chunked symm_p2p default combine (M_local=160 > max_tokens_per_rank=64 -> M_global > _symm_max_rows)
    if a2a.active_backend == "symm_mem":
        M_chunk_test = 160
        torch.manual_seed(6090)
        x_ch = torch.randn((M_chunk_test, 2560), dtype=torch.bfloat16, device=device) * 0.25
        ids_ch = torch.randint(0, 16, (M_chunk_test, 4), dtype=torch.int32, device=device)
        w_ch = torch.rand((M_chunk_test, 4), dtype=torch.float32, device=device)
        st_ch = a2a.dispatch(x_ch, ids_ch, w_ch, num_experts=16, equal_tokens=True)
        torch.manual_seed(6091 + rank)
        ep_ch = torch.randn((world_size * M_chunk_test, 2560), dtype=torch.bfloat16, device=device) * 0.25
        comb_ch = a2a.combine(ep_ch, st_ch)
        ep_all = [torch.empty_like(ep_ch) for _ in range(world_size)]
        dist.all_gather(ep_all, ep_ch, group=group)
        ref_ch = sum(t[rank * M_chunk_test : (rank + 1) * M_chunk_test].float() for t in ep_all).to(torch.bfloat16)
        assert _diff_stats(comb_ch, ref_ch)[2] >= 0.99999, "chunked default symm_p2p combine parity failed"
        # Same inputs on every rank must yield bit-identical output across ranks (0..WS-1 order)
        st_det = a2a.dispatch(x_ch, ids_ch, w_ch, num_experts=16, equal_tokens=True)
        ep_det = st_det.hidden * 0.5
        for _ in range(2):
            a2a.combine(ep_det, st_det)
        torch.cuda.synchronize(device)
        g_det = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_det):
            comb_det = a2a.combine(ep_det, st_det)
        g_det.replay()
        torch.cuda.synchronize(device)
        det_ranks = [torch.empty_like(comb_det) for _ in range(world_size)]
        dist.all_gather(det_ranks, comb_det, group=group)
        assert torch.equal(det_ranks[0], det_ranks[rank]), f"rank {rank} symm_p2p combine not bit-identical to rank 0"
        del g_det

    # Asymmetric (7 vs 0) DP-EP Qwen4_5LatentMoE forward parity (verifies 0-token rank joins A2A without deadlock)
    dp_moe.dp_equal_tokens = None
    x_asym = torch.randn((7 if rank == 0 else 0, 5120), dtype=torch.bfloat16, device=device) * 0.4
    y_asym = dp_moe(x_asym)
    assert y_asym.shape == x_asym.shape
    if rank == 0:
        y_asym_ref = ref_moe(x_asym)
        _, _, cos_asym = _diff_stats(y_asym, y_asym_ref)
        assert cos_asym > 0.996, f"Asymmetric (7 vs 0) DP-EP LatentMoE cos={cos_asym}"

    # Two-microbatch ping-pong TBO under CUDA Graph (equal_tokens=True) and eager 0-token DP rank
    dp_moe.dp_equal_tokens = True
    torch.manual_seed(6160 + rank)
    x_mb0 = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.4
    x_mb1 = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.4

    def _run_tbo_pair(t0: torch.Tensor, t1: torch.Tensor, eq_tok: bool):
        c0 = dp_moe.forward_route_and_dispatch(t0, equal_tokens=eq_tok, slot=0)
        c1 = dp_moe.forward_route_and_dispatch(t1, equal_tokens=eq_tok, slot=1)
        c0 = dp_moe.forward_routed_compute_and_combine(c0, slot=0)
        c1 = dp_moe.forward_routed_compute_and_combine(c1, slot=1)
        return dp_moe.forward_finish_combine(c0), dp_moe.forward_finish_combine(c1)

    for _ in range(3):
        _run_tbo_pair(x_mb0, x_mb1, True)
    torch.cuda.synchronize(device)
    g_tbo = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_tbo):
        y_mb0_cg, y_mb1_cg = _run_tbo_pair(x_mb0, x_mb1, True)
    nx_mb0 = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.4
    nx_mb1 = torch.randn((8, 5120), dtype=torch.bfloat16, device=device) * 0.4
    x_mb0.copy_(nx_mb0)
    x_mb1.copy_(nx_mb1)
    y_mb0_ref = ref_moe(nx_mb0)
    y_mb1_ref = ref_moe(nx_mb1)
    g_tbo.replay()
    torch.cuda.synchronize(device)
    _, _, cos_tbo0 = _diff_stats(y_mb0_cg, y_mb0_ref)
    _, _, cos_tbo1 = _diff_stats(y_mb1_cg, y_mb1_ref)
    assert cos_tbo0 > 0.996 and cos_tbo1 > 0.996, (
        f"rank={rank} TBO CG parity failed: mb0={cos_tbo0}, mb1={cos_tbo1}"
    )

    # Eager asymmetric 2-microbatch TBO with alternating 0-token DP ranks ((7 vs 0) and (0 vs 5))
    dp_moe.dp_equal_tokens = False
    xa0 = torch.randn((7 if rank == 0 else 0, 5120), dtype=torch.bfloat16, device=device) * 0.4
    xa1 = torch.randn((0 if rank == 0 else 5, 5120), dtype=torch.bfloat16, device=device) * 0.4
    ya0, ya1 = _run_tbo_pair(xa0, xa1, False)
    assert ya0.shape == xa0.shape and ya1.shape == xa1.shape
    if rank == 0:
        _, _, cos_tbo_asym0 = _diff_stats(ya0, ref_moe(xa0))
        assert cos_tbo_asym0 > 0.996, f"rank=0 TBO 0-token asym mb0 cos={cos_tbo_asym0}"
    else:
        _, _, cos_tbo_asym1 = _diff_stats(ya1, ref_moe(xa1))
        assert cos_tbo_asym1 > 0.996, f"rank=1 TBO 0-token asym mb1 cos={cos_tbo_asym1}"

    del g_mlp, g_moe, g_gnorm, g_dp, g_tbo
    if rank == 0:
        print(
            f"  PASS Distributed TP/EP={world_size} (fast_comm={ar.active_backend}, a2a={a2a.active_backend}): "
            f"DenseMLP_TP_cos={cos_tp:.6f}, LatentMoE_EP_cos={cos_ep:.6f}, "
            f"GemmaAR_CG_cos={cos_gnorm_cg:.6f}, A2A_cos={cos_a2a_r:.6f}, DP_EP_MoE({dp_backend})={cos_dp_ep:.6f}"
        )


def _shootout_dist_worker(
    rank: int, world_size: int, port: int, m_vals: tuple[int, ...], err_q: mp.Queue
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
        with torch.inference_mode():
            if rank == 0:
                bench_sm100_shootout(
                    ep_size=world_size,
                    m_values=m_vals,
                    device=device,
                    group=dist.group.WORLD,
                )
            else:
                # Non-zero ranks skip single-GPU MoE shootout and join AllReduce shootout.
                bench_sm100_shootout(
                    ep_size=world_size,
                    num_experts=world_size * 4,
                    top_k=2,
                    latent_dim=256,
                    intermediate_dim=256,
                    m_values=(),
                    device=device,
                    group=dist.group.WORLD,
                )
        err_q.put((rank, None))
    except Exception as exc:
        err_q.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        destroy_qwen4_5_all_to_all()
        destroy_qwen4_5_fast_all_reduce()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_distributed_fast_comm_tp_ep(world_size: int = 2) -> None:
    print(f"\n=== [10b] Test Distributed Qwen4_5FastAllReduce + TP/EP={world_size} + CUDA Graph ===")
    avail = torch.cuda.device_count()
    assert avail >= world_size, f"Need >= {world_size} GPUs, found {avail}"
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    err_q = ctx.Queue()
    procs = [
        ctx.Process(target=_dist_worker, args=(r, world_size, port, err_q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=150)
    errors = []
    while not err_q.empty():
        r, err = err_q.get()
        if err is not None:
            errors.append(f"rank {r}: {err}")
    for r, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            errors.append(f"rank {r}: timed out")
        elif p.exitcode not in (0, None):
            errors.append(f"rank {r}: exitcode={p.exitcode}")
    assert not errors, f"Distributed TP/EP failures: {errors}"


def _gloo_dist_worker(rank: int, world_size: int, port: int, err_q: mp.Queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        group = dist.group.WORLD
        with torch.inference_mode():
            ar = get_qwen4_5_fast_all_reduce(
                group, device=device, backend="auto", max_tokens=64, hidden_dims=(5120, 2560)
            )
            assert ar._is_gloo, "Expected _is_gloo=True on gloo group"
            torch.manual_seed(9191 + rank)
            x = torch.randn((8, 64), dtype=torch.bfloat16, device=device) * 0.3
            xr_list = [torch.empty_like(x.cpu()) for _ in range(world_size)]
            dist.all_gather(xr_list, x.cpu(), group=group)
            sum_ref = sum(t.to(device) for t in xr_list)
            xr_ar = ar.all_reduce(x)
            _, _, cos_gloo_ar = _diff_stats(xr_ar, sum_ref)
            assert cos_gloo_ar >= 0.9999, f"gloo all_reduce cos={cos_gloo_ar}"
            ag_last = ar.all_gather(x, dim=-1)
            assert torch.equal(ag_last, torch.cat([t.to(device) for t in xr_list], dim=-1))
            ag_d0 = ar.all_gather(x, dim=0)
            assert torch.equal(ag_d0, torch.cat([t.to(device) for t in xr_list], dim=0))
            streams = torch.randn((8, 4, 2560), dtype=torch.bfloat16, device=device) * 0.2
            h_post = torch.randn((8, 4), dtype=torch.float32, device=device) * 0.1
            xg = torch.randn((8, 2560), dtype=torch.bfloat16, device=device) * 0.2
            xg_list = [torch.empty_like(xg.cpu()) for _ in range(world_size)]
            dist.all_gather(xg_list, xg.cpu(), group=group)
            xg_sum = sum(t.to(device) for t in xg_list)
            hc_ref = (streams.float() + h_post.unsqueeze(-1) * xg_sum.unsqueeze(1).float()).to(torch.bfloat16)
            hc_out = ar.all_reduce_hc_combine(xg, streams, h_post)
            assert _diff_stats(hc_out, hc_ref)[2] >= 0.9995

            a2a = Qwen4_5AllToAllComm(
                group, device=device, backend="auto", max_tokens_per_rank=32, hidden_dim=2560, top_k=4
            )
            assert a2a._is_gloo
            xa = torch.randn((4, 2560), dtype=torch.bfloat16, device=device) * 0.25
            ia = torch.randint(0, 16, (4, 4), dtype=torch.int32, device=device)
            wa = torch.rand((4, 4), dtype=torch.float32, device=device)
            st_d = a2a.dispatch(xa, ia, wa, num_experts=16, equal_tokens=True)
            assert st_d.mode == "dense_gather" and st_d.hidden is not None
            comb_d = a2a.combine(st_d.hidden, st_d)
            assert _diff_stats(comb_d, (xa.float() * world_size).to(torch.bfloat16))[2] >= 0.9999

            m_z = 3 if rank == 0 else 0
            st_r = a2a.dispatch(xa[:m_z], ia[:m_z], wa[:m_z], num_experts=16, equal_tokens=False)
            assert st_r.mode == "routed_a2a", f"expected routed_a2a, got {st_r.mode}"
            local_lo, local_hi = rank * 8, (rank + 1) * 8
            in_loc = (st_r.topk_ids >= local_lo) & (st_r.topk_ids < local_hi)
            ep_out = torch.where(in_loc.any(dim=-1, keepdim=True), st_r.hidden * 2.0, torch.zeros_like(st_r.hidden))
            comb_r = a2a.combine(ep_out, st_r)
            assert comb_r.shape == (m_z, 2560), f"expected {(m_z, 2560)}, got {comb_r.shape}"
            if m_z > 0:
                owners = torch.zeros((m_z,), dtype=torch.float32, device=device)
                for r_i in range(world_size):
                    owners += ((ia[:m_z] >= r_i * 8) & (ia[:m_z] < (r_i + 1) * 8)).any(dim=-1).float()
                exp_r = (xa[:m_z].float() * 2.0 * owners[:, None]).to(torch.bfloat16)
                cos_r = _diff_stats(comb_r, exp_r)[2]
                assert cos_r >= 0.9999, f"gloo routed_a2a cos={cos_r}" 
            a2a.destroy()
            if rank == 0:
                print(f"  PASS Gloo CUDA fallback fast_comm + AllToAll (AR cos={cos_gloo_ar:.6f})")
        err_q.put((rank, None))
    except Exception as exc:
        err_q.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        destroy_qwen4_5_all_to_all()
        destroy_qwen4_5_fast_all_reduce()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_distributed_gloo_fallback(world_size: int = 2) -> None:
    print(f"\n=== [10d] Test Gloo Backend CPU-Bounce Fallback (world_size={world_size}) ===")
    avail = torch.cuda.device_count()
    assert avail >= world_size, f"Need >= {world_size} GPUs, found {avail}"
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    err_q = ctx.Queue()
    procs = [
        ctx.Process(target=_gloo_dist_worker, args=(r, world_size, port, err_q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    errors = []
    while not err_q.empty():
        r, err = err_q.get()
        if err is not None:
            errors.append(f"rank {r}: {err}")
    for r, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            errors.append(f"rank {r}: timed out")
        elif p.exitcode not in (0, None):
            errors.append(f"rank {r}: exitcode={p.exitcode}")
    assert not errors, f"Distributed Gloo fallback failures: {errors}" 


def test_sm100_fp8_linear(device: torch.device) -> None:
    print("\n=== [10c] Test SM100FP8Linear & FP8 Dense/Latent MoE Wiring ===")
    torch.manual_seed(7070)
    dev_idx = int(device.index or 0)
    for in_f, out_f in ((5120, 2560), (1280, 5120), (5120, 12288)):
        ref_lin = torch.nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16, device=device)
        torch.nn.init.normal_(ref_lin.weight, 0.0, 0.02)
        fp8_lin = SM100FP8Linear.from_linear(
            ref_lin,
            mode="auto",
            keep_bf16=(out_f < 10240),
            prepare_trtllm_ll=True,
            prepare_mxfp8=True,
            prepare_deepgemm_fp8=True,
            prepare_w4a8=True,
        )
        for m in (1, 8, 64, 512):
            x = torch.randn((m, in_f), dtype=torch.bfloat16, device=device) * 0.5
            y_ref = ref_lin(x)
            y_auto = fp8_lin(x)
            _, _, cos_a = _diff_stats(y_auto, y_ref)
            assert cos_a >= 0.999, f"SM100FP8Linear auto ({in_f}->{out_f}, M={m}) cos={cos_a}"
        # Explicit backend modes when available
        x64 = torch.randn((64, in_f), dtype=torch.bfloat16, device=device) * 0.5
        y64_ref = ref_lin(x64)
        for mname in ("scaled_mm", "trtllm_ll", "mxfp8", "deepgemm_fp8"):
            if mname == "trtllm_ll" and not has_flashinfer_trtllm_ll(dev_idx):
                continue
            if mname == "mxfp8" and not has_flashinfer_mxfp8(dev_idx):
                continue
            if mname == "deepgemm_fp8" and not has_deepgemm_fp8(dev_idx):
                continue
            y_m = fp8_lin(x64, mode=mname)
            _, _, cos_m = _diff_stats(y_m, y64_ref)
            assert cos_m >= 0.999, f"SM100FP8Linear mode={mname} cos={cos_m}"
        if fp8_lin.w_fp4_i8 is not None:
            y_w4 = fp8_lin(x64, mode="w4a8")
            _, _, cos_w4 = _diff_stats(y_w4, y64_ref)
            assert cos_w4 >= 0.99, f"SM100FP8Linear w4a8 cos={cos_w4}"
        # Pre-quant rowwise reuse
        xq, xsc = quantize_fp8_rowwise_triton(x64)
        y_pq = fp8_lin.forward_prequant_rowwise(xq, xsc, orig_shape=(64,))
        _, _, cos_pq = _diff_stats(y_pq, y64_ref)
        assert cos_pq >= 0.999, f"forward_prequant_rowwise cos={cos_pq}"

        # CUDA Graph capture/replay on Decode M=8 auto
        x_cg = torch.randn((8, in_f), dtype=torch.bfloat16, device=device) * 0.5
        y_cg_ref = ref_lin(x_cg)
        for _ in range(2):
            fp8_lin(x_cg)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_cg = fp8_lin(x_cg)
        g.replay()
        torch.cuda.synchronize(device)
        _, _, cos_cg = _diff_stats(y_cg, y_cg_ref)
        assert cos_cg >= 0.999, f"SM100FP8Linear CG cos={cos_cg}"
        del g

    # Verify SM100/SM103 Pareto frontier dispatch on keep_bf16=True (M*K*N >= 1e11 + allow_decode_scaled_mm)
    assert isinstance(has_flashinfer_mxfp8(dev_idx), bool)
    assert isinstance(has_flashinfer_trtllm_ll(dev_idx), bool)
    narrow_lin = SM100FP8Linear(5120, 2560, mode="auto", keep_bf16=True, device=device)
    torch.nn.init.normal_(narrow_lin.weight, 0.0, 0.02)
    narrow_lin.prepare_weights()
    assert narrow_lin._resolve_backend(8, device) == "bf16"
    assert narrow_lin._resolve_backend(512, device) == "bf16"
    assert narrow_lin._resolve_backend(2048, device) == "bf16"
    assert narrow_lin._resolve_backend(4096, device) == "bf16"
    assert narrow_lin._resolve_backend(8192, device) == (
        "mxfp8" if has_flashinfer_mxfp8(dev_idx) else "bf16"
    )
    if has_flashinfer_mxfp8(dev_idx):
        import flashinfer  # type: ignore

        xp = torch.randn((64, 5120), dtype=torch.bfloat16, device=device) * 0.5
        ax, asc = flashinfer.mxfp8_quantize(xp, is_sf_swizzled_layout=True, alignment=32)
        yp_pq = narrow_lin.forward_prequant_mxfp8(ax, asc, orig_shape=(64,))
        assert _diff_stats(yp_pq, F.linear(xp, narrow_lin.weight))[2] >= 0.999

    sh_dn_lin = SM100FP8Linear(1280, 5120, mode="auto", keep_bf16=True, device=device)
    torch.nn.init.normal_(sh_dn_lin.weight, 0.0, 0.02)
    sh_dn_lin.prepare_weights()
    assert sh_dn_lin._resolve_backend(8192, device) == "bf16"
    assert sh_dn_lin._resolve_backend(20480, device) == (
        "mxfp8" if has_flashinfer_mxfp8(dev_idx) else "bf16"
    )

    tp4_sh_dn = SM100FP8Linear(320, 5120, mode="auto", keep_bf16=True, device=device)
    torch.nn.init.normal_(tp4_sh_dn.weight, 0.0, 0.02)
    tp4_sh_dn.prepare_weights()
    assert tp4_sh_dn._resolve_backend(20480, device) == "bf16"

    # Non-lm_head wide layers (DenseMLP.gate_up=25600, GDN.qkvz=40960) stay BF16 during Decode.
    for wide_out in (25600, 40960):
        non_lm_wide = SM100FP8Linear(
            5120, wide_out, mode="auto", keep_bf16=True, allow_decode_scaled_mm=False, device=device
        )
        torch.nn.init.normal_(non_lm_wide.weight, 0.0, 0.02)
        non_lm_wide.prepare_weights()
        for m_w in (1, 8, 64, 512):
            assert non_lm_wide._resolve_backend(m_w, device) == "bf16"
        del non_lm_wide

    dflash_fc_lin = SM100FP8Linear(
        25600, 5120, mode="auto", keep_bf16=True, prepare_trtllm_ll=False, prepare_mxfp8=False, device=device
    )
    torch.nn.init.normal_(dflash_fc_lin.weight, 0.0, 0.02)
    dflash_fc_lin.prepare_weights()
    for m_df in (1, 8, 64, 512):
        assert dflash_fc_lin._resolve_backend(m_df, device) == "bf16"
    wide_lin = SM100FP8Linear(
        5120,
        62080,
        mode="auto",
        keep_bf16=True,
        prepare_trtllm_ll=True,
        prepare_mxfp8=True,
        allow_decode_scaled_mm=True,
        device=device,
    )
    torch.nn.init.normal_(wide_lin.weight, 0.0, 0.02)
    wide_lin.prepare_weights()
    for m_w in (8, 64, 512):
        assert wide_lin._resolve_backend(m_w, device) == "scaled_mm"
        xw = torch.randn((m_w, 5120), dtype=torch.bfloat16, device=device) * 0.5
        yw_ref = F.linear(xw, wide_lin.weight)
        yw_auto = wide_lin(xw)
        assert _diff_stats(yw_auto, yw_ref)[2] >= 0.999, f"wide lm_head M={m_w} cos low"
    assert wide_lin._resolve_backend(2048, device) == (
        "mxfp8" if has_flashinfer_mxfp8(dev_idx) else "scaled_mm"
    )
    prev_pref_ll = os.environ.get("QWEN4_5_FP8_PREFER_TRTLLM_LL")
    try:
        os.environ["QWEN4_5_FP8_PREFER_TRTLLM_LL"] = "1"
        if has_flashinfer_trtllm_ll(dev_idx):
            assert wide_lin._resolve_backend(8, device) == "trtllm_ll"
        assert wide_lin._resolve_backend(64, device) == "scaled_mm"
    finally:
        if prev_pref_ll is None:
            os.environ.pop("QWEN4_5_FP8_PREFER_TRTLLM_LL", None)
        else:
            os.environ["QWEN4_5_FP8_PREFER_TRTLLM_LL"] = prev_pref_ll
    del narrow_lin, sh_dn_lin, tp4_sh_dn, dflash_fc_lin, wide_lin

    # DenseMLP TP=1 (N=25600) & TP=4 dense_linear_mode="auto" parity (must keep cos >= 0.9999)
    cfg = Qwen4_5ArchConfig()
    for tp_sz in (1, 4):
        mlp_bf16 = Qwen4_5DenseMLP.from_config(cfg, tp_size=tp_sz, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            for p in mlp_bf16.parameters():
                p.normal_(0.0, 0.02)
        mlp_fp8 = Qwen4_5DenseMLP.from_config(
            cfg, tp_size=tp_sz, dense_linear_mode="auto", dtype=torch.bfloat16, device=device
        )
        mlp_fp8.load_state_dict(mlp_bf16.state_dict())
        mlp_bf16.prepare_weights(fuse_gate_up=True)
        mlp_fp8.prepare_weights(fuse_gate_up=True)
        assert mlp_fp8._fp8_gate_up is not None
        assert mlp_fp8._fp8_gate_up._resolve_backend(64, device) == "bf16"
        assert mlp_fp8._fp8_gate_up._resolve_backend(4096, device) == "bf16"
        for m in (8, 64, 512):
            xm = torch.randn((m, 5120), dtype=torch.bfloat16, device=device) * 0.5
            ym_ref = mlp_bf16(xm)
            ym_fp8 = mlp_fp8(xm)
            _, _, cos_mlp = _diff_stats(ym_fp8, ym_ref)
            assert cos_mlp >= 0.9999, f"Qwen4_5DenseMLP TP={tp_sz} auto M={m} cos={cos_mlp}"
        del mlp_bf16, mlp_fp8

    # LatentMoE fused W_3x + [2, H] epilogue + 2-pass norm (Decode, Prefill M=4096, TBO, CG, L=5120)
    moe_bf16 = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        dtype=torch.bfloat16,
        device=device,
        enable_shared_overlap=False,
        backend="triton_splitk",
    )
    _init_latent_moe_weights(moe_bf16, seed=8080)
    moe_bf16.prepare_weights()
    moe_fp8 = Qwen4_5LatentMoE(
        hidden_size=5120,
        routed_expert_hidden_size=2560,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=1280,
        num_experts=16,
        num_experts_per_tok=4,
        dtype=torch.bfloat16,
        device=device,
        enable_shared_overlap=True,
        dense_linear_mode="auto",
        backend="triton_splitk",
    )
    moe_fp8.load_state_dict(moe_bf16.state_dict(), strict=True)
    moe_fp8.prepare_weights()
    assert moe_fp8._latent_down_gate_shared_gu_weight is not None
    assert moe_fp8._latent_shared_gates_weight is not None
    # Amplify router weight margin so tiny FP8 noise does not flip random top-k ties.
    with torch.no_grad():
        moe_bf16.gate.weight.mul_(4.0)
        moe_fp8.gate.weight.mul_(4.0)
    torch.manual_seed(8081)
    for m_chk, min_cos in ((8, 0.9999), (64, 0.9999), (4096, 0.9999), (8192, 0.999)):
        xm_moe = torch.randn((m_chk, 5120), dtype=torch.bfloat16, device=device) * 0.4
        ym_auto = moe_fp8(xm_moe)
        ym_bf = moe_bf16(xm_moe)
        _, _, cos_moe_fp8 = _diff_stats(ym_auto, ym_bf)
        assert cos_moe_fp8 >= min_cos, f"Qwen4_5LatentMoE FP8 auto M={m_chk} cos={cos_moe_fp8}"
    # Verify CUDA Graph capture/replay on LatentMoE R4 path
    xm_cg = torch.randn((64, 5120), dtype=torch.bfloat16, device=device) * 0.4
    ym_cg_ref = moe_bf16(xm_cg)
    for _ in range(2):
        moe_fp8(xm_cg)
    torch.cuda.synchronize(device)
    g_moe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_moe):
        ym_cg_out = moe_fp8(xm_cg)
    g_moe.replay()
    torch.cuda.synchronize(device)
    assert _diff_stats(ym_cg_out, ym_cg_ref)[2] >= 0.9999
    del g_moe
    # Verify TBO 3-stage path matches standard forward
    ctx_tbo = moe_fp8.forward_route_and_dispatch(xm_cg)
    ctx_tbo = moe_fp8.forward_routed_compute_and_combine(ctx_tbo)
    ym_tbo = moe_fp8.forward_finish_combine(ctx_tbo)
    assert _diff_stats(ym_tbo, moe_fp8(xm_cg))[2] >= 0.9999

    # Verify 2-pass norm works with L > 4096 (5120) and shared_expert_gate=False
    moe_big_l = Qwen4_5LatentMoE(
        hidden_size=512,
        routed_expert_hidden_size=5120,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        shared_expert_gate=False,
        num_experts=8,
        num_experts_per_tok=2,
        dtype=torch.bfloat16,
        device=device,
        enable_shared_overlap=False,
        backend="triton_splitk",
    )
    _init_latent_moe_weights(moe_big_l, seed=8082)
    moe_big_l.prepare_weights()
    xb = torch.randn((16, 512), dtype=torch.bfloat16, device=device) * 0.4
    yb_tri = moe_big_l(xb, use_triton_mxfp4=True)
    yb_ref = moe_big_l(xb, use_triton_mxfp4=False)
    assert _diff_stats(yb_tri, yb_ref)[2] >= 0.9999
    del moe_bf16, moe_fp8, moe_big_l
    print(f"  PASS SM100FP8Linear & FP8 Dense/LatentMoE wiring (LatentMoE M=8192 cos={cos_moe_fp8:.6f})")


def test_r4_combine_bitwise_schedule_and_multigpu(device: torch.device) -> None:
    print("[10d] Testing R4 combine bitwise parity, fused Decode schedule & multi-GPU device ctx...")
    from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
        _build_compact_decode_schedule,
        _gpu_expert_bincount,
    )

    torch.manual_seed(20261126)
    # 1. Bitwise BF16 parity on _combine_latent_and_shared (avoids fma.rn.bf16 contraction)
    for has_sg in (True, False):
        for H in (512, 5120):
            moe = Qwen4_5LatentMoE(
                hidden_size=H,
                routed_expert_hidden_size=min(H, 1280),
                moe_intermediate_size=128,
                shared_expert_intermediate_size=128,
                shared_expert_gate=has_sg,
                num_experts=8,
                num_experts_per_tok=2,
                dtype=torch.bfloat16,
                device=device,
                enable_shared_overlap=False,
            )
            _init_latent_moe_weights(moe, seed=9001 + H + int(has_sg))
            moe.prepare_weights()
            for M in (1, 7, 64, 512):
                x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
                r = torch.randn((M, H), dtype=torch.bfloat16, device=device)
                s = torch.randn((M, H), dtype=torch.bfloat16, device=device)
                y_tri = moe._combine_latent_and_shared(x, r, s, run_triton=True)
                y_ref = moe._combine_latent_and_shared(x, r, s, run_triton=False)
                assert torch.equal(y_tri, y_ref), (
                    f"_combine_latent_and_shared bitwise diff has_sg={has_sg} H={H} M={M} "
                    f"n_diffs={(y_tri != y_ref).sum().item()}"
                )
            del moe

    # 2. Fused _build_compact_decode_schedule parity (CPU + GPU, ep_size=1 & 4) + CG replay
    for dev in (torch.device("cpu"), device):
        for num_experts, ep_size, ep_start in ((8, 1, 0), (256, 1, 0), (64, 4, 64)):
            for M in (1, 2, 7, 64):
                for w_dtype in (torch.bfloat16, torch.float32):
                    for id_dtype in (torch.int32, torch.int64):
                        total_e = num_experts * ep_size
                        ids = torch.stack(
                            [torch.randperm(total_e, device=dev)[:8] for _ in range(M)]
                        ).to(id_dtype)
                        w = torch.randn((M, 8), device=dev, dtype=w_dtype)
                        sched = _build_compact_decode_schedule(
                            ids, w, num_experts, 16, ep_start=ep_start, ep_size=ep_size
                        )
                        flat_ids = ids.reshape(-1)
                        flat_w = w.float().reshape(-1)
                        slot_idx = torch.arange(M * 8, dtype=torch.int32, device=dev)
                        if ep_size > 1:
                            in_range = (flat_ids >= ep_start) & (flat_ids < ep_start + num_experts)
                            flat_local = torch.where(
                                in_range,
                                flat_ids - ep_start,
                                torch.full_like(flat_ids, num_experts),
                            )
                            _, perm_i64 = torch.sort(flat_local, stable=True)
                            slot_vals = torch.where(
                                in_range.index_select(0, perm_i64),
                                slot_idx,
                                torch.full_like(slot_idx, -1),
                            )
                            counts = _gpu_expert_bincount(flat_local, num_experts)
                        else:
                            _, perm_i64 = torch.sort(flat_ids, stable=True)
                            slot_vals = slot_idx
                            counts = _gpu_expert_bincount(flat_ids, num_experts)
                        ref_tok = (perm_i64 // 8).to(torch.int32)
                        ref_w = flat_w.index_select(0, perm_i64)
                        ref_inv = torch.empty((M * 8,), dtype=torch.int32, device=dev)
                        ref_inv.index_copy_(0, perm_i64, slot_vals)
                        ref_inv = ref_inv.reshape(M, 8)
                        n_act = int(sched.num_active_slots_buf.item())
                        n_tiles = int(sched.num_active_tiles_buf.item())
                        assert n_act == int(counts.sum().item())
                        assert int(sched.tile_m_count[:n_tiles].sum().item()) == n_act
                        if M > 1 or dev.type == "cpu":
                            assert torch.equal(sched.inv_slots, ref_inv)
                            assert torch.equal(sched.sorted_tok_ids, ref_tok)
                            assert torch.equal(sched.sorted_weights, ref_w)

    cg_ids = torch.stack([torch.randperm(256, device=device)[:8] for _ in range(64)]).to(torch.int32)
    cg_w = torch.randn((64, 8), device=device, dtype=torch.bfloat16)
    _build_compact_decode_schedule(cg_ids, cg_w, 256, 16)
    torch.cuda.synchronize(device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cg_out = _build_compact_decode_schedule(cg_ids, cg_w, 256, 16)
    new_ids = torch.stack([torch.randperm(256, device=device)[:8] for _ in range(64)]).to(torch.int32)
    new_w = torch.randn((64, 8), device=device, dtype=torch.bfloat16)
    cg_ids.copy_(new_ids)
    cg_w.copy_(new_w)
    g.replay()
    ref_cg = _build_compact_decode_schedule(new_ids, new_w, 256, 16)
    assert torch.equal(cg_out.inv_slots, ref_cg.inv_slots)
    assert torch.equal(cg_out.sorted_weights, ref_cg.sorted_weights)
    del g, cg_out, ref_cg

    # 3. Single-process multi-GPU guard: execute on cuda:1 while current_device() == 0
    if torch.cuda.device_count() >= 2:
        torch.cuda.set_device(0)
        dev1 = torch.device("cuda:1")
        lin = SM100FP8Linear(
            256,
            256,
            mode="scaled_mm",
            keep_bf16=True,
            allow_decode_scaled_mm=True,
            device=dev1,
        )
        torch.nn.init.normal_(lin.weight, 0.0, 0.02)
        lin.prepare_weights()
        x1 = torch.randn((16, 256), dtype=torch.bfloat16, device=dev1)
        y1 = lin(x1)
        assert y1.device == dev1
        act_y = SituAndMul()(torch.randn((16, 512), dtype=torch.bfloat16, device=dev1))
        assert act_y.device == dev1
        moe1 = Qwen4_5LatentMoE(
            hidden_size=256,
            routed_expert_hidden_size=128,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            dtype=torch.bfloat16,
            device=dev1,
            enable_shared_overlap=False,
        )
        _init_latent_moe_weights(moe1, seed=9901)
        moe1.prepare_weights()
        m1_out = moe1(torch.randn((8, 256), dtype=torch.bfloat16, device=dev1), use_triton_mxfp4=True)
        assert m1_out.device == dev1
        del lin, x1, y1, act_y, moe1, m1_out
    print("  PASS R4 combine bitwise parity, fused Decode schedule & multi-GPU device ctx")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen 4.5-Flash LatentMoE & SM100 MXFP4 test suite")
    parser.add_argument("--distributed", action="store_true", help="Run multi-GPU FastAllReduce/TP/EP tests")
    parser.add_argument("--world-size", type=int, default=2, help="World size for --distributed")
    parser.add_argument("--shootout", action="store_true", help="Run SM100 MoE shootout benchmark")
    parser.add_argument("--shootout-ep", type=int, default=4, help="EP size for --shootout")
    parser.add_argument("--shootout-m", type=str, default="1,4,8,16,32,64,512,2048,5120,16384,20480", help="Comma-separated M values")
    args, _ = parser.parse_known_args()

    assert torch.cuda.is_available(), "CUDA GPU is required"
    device = torch.device("cuda:0")
    print(
        f"Running Qwen 4.5-Flash Latent MoE & MXFP4 tests on "
        f"{torch.cuda.get_device_name(device)} (SM{torch.cuda.get_device_capability(device)})"
    )
    if args.shootout:
        m_vals = tuple(int(x.strip()) for x in args.shootout_m.split(",") if x.strip())
        if args.distributed:
            port = _find_free_port()
            ctx = mp.get_context("spawn")
            err_q = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_shootout_dist_worker,
                    args=(r, args.world_size, port, m_vals, err_q),
                )
                for r in range(args.world_size)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=300)
            errors = []
            while not err_q.empty():
                r, err = err_q.get()
                if err is not None:
                    errors.append(f"rank {r}: {err}")
            for r, p in enumerate(procs):
                if p.is_alive():
                    p.terminate()
                    errors.append(f"rank {r}: timed out")
                elif p.exitcode not in (0, None):
                    errors.append(f"rank {r}: exitcode={p.exitcode}")
            assert not errors, f"Distributed shootout errors: {errors}"
            return 0
        with torch.inference_mode():
            bench_sm100_shootout(ep_size=args.shootout_ep, m_values=m_vals, device=device)
        return 0

    with torch.inference_mode():
        test_situ_and_mul(device)
        test_gemma_rmsnorm(device)
        test_mxfp4_format_and_dot_scaled(device)
        test_dense_mlp(device)
        test_latent_moe_small(device)
        test_and_benchmark_latent_moe_real(device)
        test_situ_glu_fp8_quant(device)
        test_sm100_mxfp4_backends_and_cuda_graph(device)
        test_dense_mlp_tp_simulated(device)
        test_sm100_fp8_linear(device)
        test_r4_combine_bitwise_schedule_and_multigpu(device)

    if args.distributed:
        test_distributed_fast_comm_tp_ep(world_size=args.world_size)
        test_distributed_gloo_fallback(world_size=args.world_size)

    print("\nALL TESTS AND BENCHMARKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
