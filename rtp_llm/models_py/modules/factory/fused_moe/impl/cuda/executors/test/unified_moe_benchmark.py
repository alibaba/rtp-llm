"""Unified FP4 vs FP8 MoE Benchmark on SM100.

All implementations run the SAME full MoE forward: FC1 + SiLU + FC2.
Same N=2048 (intermediate), K=7168 (hidden) for all.

Implementations compared:
  FP4-CD:      CuteDSL FP4 (FlashInfer CuteDSL JIT, masked 3D)
  FP4-TRT:     TRT-LLM FP4 fused MoE (FlashInfer/TRT-LLM, end-to-end fused)
  FP4-CUTLASS: vLLM/SGLang CUTLASS FP4 (standalone CUTLASS GemmUniversal)
  FP8-FI:      FlashInfer FP8 groupwise (float32 blockwise scale)
  FP8-DGM:     DeepGEMM FP8 masked (UE8M0 scale, masked 3D)
  FP8-DGC:     DeepGEMM FP8 contiguous (UE8M0 scale, contiguous)
  FP8-CUTLASS: CUTLASS FP8 per-tensor (rtp_kernel)
  FP8-TRT:     TRT-LLM FP8 fused MoE (FlashInfer/TRT-LLM, end-to-end fused)

Scenarios include both uniform and non-uniform expert activation patterns.
"""
import time
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None

from rtp_llm.utils.model_weight import W

BLOCK_SIZE = 128
WARMUP_ITERS = 10
BENCH_ITERS = 50
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
SEED = 42

# Collect autotune details for all kernels (printed at end of benchmark)
_AUTOTUNE_LOG = []


def _bench_time(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - start) / iters) * 1000


def _compute_moe_tflops(total_tokens, N, K, avg_ms):
    """Full MoE: FC1([M,K]x[2N,K]^T) + FC2([M,N]x[K,N]^T)"""
    flops = total_tokens * (2 * N) * K * 2 + total_tokens * K * N * 2
    return (flops / (avg_ms / 1000)) / 1e12


def _safe_run(fn):
    try:
        return fn()
    except Exception as e:
        return None, None, str(e)[:200]


# =========================================================================
# 1. CuteDSL FP4 (Full MoE via Executor, masked 3D layout)
# =========================================================================

def _bench_cutedsl_fp4(E, tokens_per_expert, K, N):
    """tokens_per_expert: list[int] of length E, or int for uniform."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import CutedslFp4Executor
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertForwardPayload, ExpertTokensMetadata
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig
    from flashinfer import scaled_fp4_grouped_quantize

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    max_m = max(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)

    w1_amax = w1_bf16.abs().amax(dim=(1, 2)).float()
    w2_amax = w2_bf16.abs().amax(dim=(1, 2)).float()
    w1_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
    w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

    w1_fp4, w1_bs = scaled_fp4_grouped_quantize(
        w1_bf16, torch.full((E,), 2 * N, dtype=torch.int32, device=device), w1_gs)
    w2_fp4, w2_bs = scaled_fp4_grouped_quantize(
        w2_bf16, torch.full((E,), K, dtype=torch.int32, device=device), w2_gs)

    weights = {
        W.moe_w1: w1_fp4.permute(2, 0, 1), W.moe_w2: w2_fp4.permute(2, 0, 1),
        W.moe_s1: w1_bs, W.moe_s2: w2_bs,
        W.moe_w1_s2: 1.0 / w1_gs, W.moe_w2_s2: 1.0 / w2_gs,
        W.moe_w1_i_s: torch.ones(E, dtype=torch.float32, device=device),
        W.moe_w2_i_s: torch.ones(E, dtype=torch.float32, device=device),
    }

    config = _make_config(E, K, N, 8, max_m)
    executor = CutedslFp4Executor(config, FusedMoEQuantConfig(
        quant_dtype=torch.uint8, per_act_token_quant=False, per_out_ch_quant=False, block_shape=[16, 16]), weights)

    hidden = torch.randn(E, max_m, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1
    masked_m = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device)
    payload = ExpertForwardPayload(
        expert_x=hidden, expert_x_origin_dtype=torch.bfloat16, expert_x_scale=None,
        expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=masked_m, expert_num_tokens_cpu=None))

    def run():
        return executor.execute(payload, "silu", None, None, False, None)

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 2. FlashInfer FP8 Groupwise (Full MoE: quant+FC1+act+FC2)
# =========================================================================

def _per_block_quantize_fp8(tensor, block_size=128):
    has_batch = tensor.dim() == 3
    if has_batch:
        E_dim, N_dim, K_dim = tensor.shape
        flat = tensor.reshape(-1, K_dim).float()
    else:
        N_dim, K_dim = tensor.shape
        flat = tensor.float()
    N_total = flat.shape[0]
    n_blocks = (N_total + block_size - 1) // block_size
    k_blocks = (K_dim + block_size - 1) // block_size
    N_pad = n_blocks * block_size
    K_pad = k_blocks * block_size
    padded = torch.zeros(N_pad, K_pad, device=tensor.device, dtype=torch.float32)
    padded[:N_total, :K_dim] = flat
    viewed = padded.view(n_blocks, block_size, k_blocks, block_size)
    amax = viewed.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = amax / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    scale_exp = scale.unsqueeze(1).unsqueeze(3).expand_as(viewed)
    quantized = (viewed / scale_exp).reshape(N_pad, K_pad)
    fp8 = quantized[:N_total, :K_dim].to(torch.float8_e4m3fn)
    if has_batch:
        fp8 = fp8.view(E_dim, N_dim, K_dim)
        scale = scale.view(E_dim, N_dim // block_size if N_dim % block_size == 0 else n_blocks // E_dim, k_blocks)
    return fp8, scale


def _bench_flashinfer_fp8(E, tokens_per_expert, K, N):
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import _recompute_float32_scales
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    M_padded = max(((m + 3) // 4) * 4 for m in tokens_per_expert)
    total_M = M_padded * E
    total_tokens = sum(tokens_per_expert)

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)
    w1_fp8, _ = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, _ = _per_block_quantize_fp8(w2_bf16)
    w1_scale_mn = _recompute_float32_scales(w1_fp8).permute(0, 2, 1).contiguous()
    w2_scale_mn = _recompute_float32_scales(w2_fp8).permute(0, 2, 1).contiguous()

    grouped_input = torch.randn(total_M, K, device=device, dtype=torch.bfloat16,
                                 generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1
    m_indptr = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        m_indptr[i + 1] = m_indptr[i] + M_padded

    def make_run(mma_sm):
        def run():
            inp_fp8, inp_scale = sgl_per_token_group_quant_fp8(
                grouped_input, group_size=BLOCK_SIZE,
                column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False)
            inp_scale_mn = inp_scale.T.contiguous()
            fc1 = group_gemm_fp8_nt_groupwise(
                a=inp_fp8, b=w1_fp8, a_scale=inp_scale_mn, b_scale=w1_scale_mn,
                m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16,
                mma_sm=mma_sm)
            act = torch.empty((total_M, N), device=device, dtype=torch.bfloat16)
            silu_and_mul(act, fc1)
            fc2_fp8, fc2_scale = sgl_per_token_group_quant_fp8(
                act, group_size=BLOCK_SIZE,
                column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False)
            fc2_scale_mn = fc2_scale.T.contiguous()
            return group_gemm_fp8_nt_groupwise(
                a=fc2_fp8, b=w2_fp8, a_scale=fc2_scale_mn, b_scale=w2_scale_mn,
                m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16,
                mma_sm=mma_sm)
        return run

    # Exhaustive autotune: benchmark both mma_sm=1 (128x128) and mma_sm=2 (256x128)
    best_ms, best_sm = float('inf'), 1
    tune_details = []
    for mma_sm in [1, 2]:
        try:
            ms = _bench_time(make_run(mma_sm))
            tune_details.append(f"sm{mma_sm}={ms:.3f}ms")
            if ms < best_ms:
                best_ms, best_sm = ms, mma_sm
        except Exception:
            tune_details.append(f"sm{mma_sm}=ERR")
    _AUTOTUNE_LOG.append(f"FP8-FI totM={total_tokens}: {' '.join(tune_details)} → best=sm{best_sm}")

    avg_ms = best_ms
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 3. DeepGEMM FP8 Masked (Full MoE: quant+FC1+act+requant+FC2)
# =========================================================================

def _bench_deepgemm_masked(E, tokens_per_expert, K, N):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_masked
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    max_m_raw = max(tokens_per_expert)
    max_m = align(max_m_raw, BLOCK_SIZE)
    total_tokens = sum(tokens_per_expert)

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)

    w1_fp8_data = torch.empty(E, 2 * N, K, device=device, dtype=torch.float8_e4m3fn)
    w1_fp8_scale = torch.empty(E, ceil_div(2 * N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    w2_fp8_data = torch.empty(E, K, N, device=device, dtype=torch.float8_e4m3fn)
    w2_fp8_scale = torch.empty(E, ceil_div(K, 128), ceil_div(N, 128), device=device, dtype=torch.float32)
    for i in range(E):
        w1_fp8_data[i], w1_fp8_scale[i] = per_block_cast_to_fp8(w1_bf16[i], use_ue8m0=True)
        w2_fp8_data[i], w2_fp8_scale[i] = per_block_cast_to_fp8(w2_bf16[i], use_ue8m0=True)
    w1_fp8 = (w1_fp8_data, w1_fp8_scale)
    w2_fp8 = (w2_fp8_data, w2_fp8_scale)

    a_bf16 = torch.randn(E, max_m, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1
    a_fp8_data = torch.empty(E, max_m, K, device=device, dtype=torch.float8_e4m3fn)
    a_fp8_scale = torch.empty(E, max_m, ceil_div(K, 128), device=device, dtype=torch.float32)
    for i in range(E):
        a_fp8_data[i], a_fp8_scale[i] = per_token_cast_to_fp8(a_bf16[i], use_ue8m0=True)
    a_fp8 = (a_fp8_data, a_fp8_scale)

    masked_m = torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)
    fc1_out = torch.empty(E, max_m, 2 * N, device=device, dtype=torch.bfloat16)
    fc2_out = torch.empty(E, max_m, K, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_masked(a_fp8, w1_fp8, fc1_out, masked_m, max_m_raw)
        act = torch.empty(E, max_m, N, device=device, dtype=torch.bfloat16)
        silu_and_mul(act.view(-1, N), fc1_out.view(-1, 2 * N))
        act_fp8_data = torch.empty(E, max_m, N, device=device, dtype=torch.float8_e4m3fn)
        act_fp8_scale = torch.empty(E, max_m, ceil_div(N, 128), device=device, dtype=torch.float32)
        for i in range(E):
            act_fp8_data[i], act_fp8_scale[i] = per_token_cast_to_fp8(act[i], use_ue8m0=True)
        m_grouped_fp8_gemm_nt_masked((act_fp8_data, act_fp8_scale), w2_fp8, fc2_out, masked_m, max_m_raw)
        return fc2_out

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 4. DeepGEMM FP8 Contiguous (Full MoE: quant+FC1+act+requant+FC2)
# =========================================================================

def _bench_deepgemm_contiguous(E, tokens_per_expert, K, N):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_contiguous
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    mk_align = 128
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    total_tokens = sum(tokens_per_expert)

    # Align each expert's token count
    aligned_counts = [align(m, mk_align) for m in tokens_per_expert]
    total_M = sum(aligned_counts)

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32, generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)

    w1_fp8_data = torch.empty(E, 2 * N, K, device=device, dtype=torch.float8_e4m3fn)
    w1_fp8_scale = torch.empty(E, ceil_div(2 * N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    w2_fp8_data = torch.empty(E, K, N, device=device, dtype=torch.float8_e4m3fn)
    w2_fp8_scale = torch.empty(E, ceil_div(K, 128), ceil_div(N, 128), device=device, dtype=torch.float32)
    for i in range(E):
        w1_fp8_data[i], w1_fp8_scale[i] = per_block_cast_to_fp8(w1_bf16[i], use_ue8m0=True)
        w2_fp8_data[i], w2_fp8_scale[i] = per_block_cast_to_fp8(w2_bf16[i], use_ue8m0=True)
    w1_fp8 = (w1_fp8_data, w1_fp8_scale)
    w2_fp8 = (w2_fp8_data, w2_fp8_scale)

    a_bf16 = torch.randn(total_M, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1
    a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=True)

    # m_indices: expert id per row
    m_indices = torch.empty(total_M, device=device, dtype=torch.int32)
    offset = 0
    for i in range(E):
        m_indices[offset:offset + tokens_per_expert[i]] = i
        m_indices[offset + tokens_per_expert[i]:offset + aligned_counts[i]] = -1
        offset += aligned_counts[i]

    fc1_out = torch.empty(total_M, 2 * N, device=device, dtype=torch.bfloat16)
    fc2_out = torch.empty(total_M, K, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_contiguous(a_fp8, w1_fp8, fc1_out, m_indices)
        act = torch.empty(total_M, N, device=device, dtype=torch.bfloat16)
        silu_and_mul(act, fc1_out)
        act_fp8 = per_token_cast_to_fp8(act, use_ue8m0=True)
        m_grouped_fp8_gemm_nt_contiguous(act_fp8, w2_fp8, fc2_out, m_indices)
        return fc2_out

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 5. TRT-LLM FP4 Fused MoE (via trtllm_fp4_block_scale_moe)
# =========================================================================

def _bench_trtllm_fp4(E, tokens_per_expert, K, N):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    total_tokens = sum(tokens_per_expert)
    top_k = min(8, E - 1)  # TRT-LLM requires num_experts > top_k

    # Generate weights
    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32,
                            generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32,
                            generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)

    # Quantize weights to FP4
    sf_vec_size = 16
    w1_global_sf = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w1_bf16.float().abs().amax(dim=(1, 2)).clamp(min=1e-4)
    w2_global_sf = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w2_bf16.float().abs().amax(dim=(1, 2)).clamp(min=1e-4)

    w1_fp4_list, w1_sf_list = [], []
    w2_fp4_list, w2_sf_list = [], []
    permute_cache = {}
    for i in range(E):
        q, sf = fp4_quantize(w1_bf16[i], w1_global_sf[i], sf_vec_size=sf_vec_size,
                             sf_use_ue8m0=False, is_sf_swizzled_layout=False)
        perm = _maybe_get_cached_w3_w1_permute_indices(permute_cache, q, epilogue_tile_m=128)
        q = q[perm]
        sf = block_scale_interleave(sf[perm])
        w1_fp4_list.append(q)
        w1_sf_list.append(sf)

        q2, sf2 = fp4_quantize(w2_bf16[i], w2_global_sf[i], sf_vec_size=sf_vec_size,
                                sf_use_ue8m0=False, is_sf_swizzled_layout=False)
        perm2 = get_w2_permute_indices_with_cache(permute_cache, q2, epilogue_tile_m=128)
        q2 = q2[perm2]
        sf2 = block_scale_interleave(sf2[perm2])
        w2_fp4_list.append(q2)
        w2_sf_list.append(sf2)

    w1_fp4 = torch.stack(w1_fp4_list)
    w1_sf = torch.stack(w1_sf_list).view(torch.float8_e4m3fn)  # kernel requires fp8 dtype
    w2_fp4 = torch.stack(w2_fp4_list)
    w2_sf = torch.stack(w2_sf_list).view(torch.float8_e4m3fn)  # kernel requires fp8 dtype

    # Input
    hidden = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1
    hs_global_sf = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / hidden.float().abs().max().clamp(min=1e-4)
    hs_fp4, hs_sf = fp4_quantize(hidden, hs_global_sf, sf_vec_size=sf_vec_size,
                                  sf_use_ue8m0=False, is_sf_swizzled_layout=False)
    # Convert scale to float8_e4m3fn view (kernel requirement)
    hs_sf = hs_sf.view(torch.float8_e4m3fn).reshape(total_tokens, -1)

    # Construct routing logits that produce desired token distribution
    routing_logits = torch.full((total_tokens, E), -10.0, device=device, dtype=torch.bfloat16)
    offset = 0
    for expert_id in range(E):
        for _ in range(tokens_per_expert[expert_id]):
            if offset < total_tokens:
                routing_logits[offset, expert_id] = 10.0
                offset += 1

    # Approximate output1 scales
    inv_w1_gs = 1.0 / w1_global_sf
    inv_hs_gs = 1.0 / hs_global_sf
    c_global_sf_approx = torch.ones(E, device=device, dtype=torch.float32)
    scale_c_fc1 = c_global_sf_approx * inv_w1_gs * inv_hs_gs
    scale_gate_fc1 = inv_w1_gs * inv_hs_gs
    scale_c_fc2 = (1.0 / c_global_sf_approx) * (1.0 / w2_global_sf)

    def run():
        return trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_sf,
            gemm1_weights=w1_fp4,
            gemm1_weights_scale=w1_sf,
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=w2_fp4,
            gemm2_weights_scale=w2_sf,
            gemm2_bias=None,
            output1_scale_scalar=scale_c_fc1,
            output1_scale_gate_scalar=scale_gate_fc1,
            output2_scale_scalar=scale_c_fc2,
            num_experts=E, top_k=top_k,
            n_group=None, topk_group=None,
            intermediate_size=N,
            local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,  # Renormalize: TopK -> Softmax
            do_finalize=True,
            tune_max_num_tokens=max(total_tokens, 4096),
        )

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 6. TRT-LLM FP8 Fused MoE (via trtllm_fp8_block_scale_moe)
# =========================================================================

def _bench_trtllm_fp8(E, tokens_per_expert, K, N):
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    total_tokens = sum(tokens_per_expert)
    top_k = min(8, E - 1)  # TRT-LLM requires num_experts > top_k

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32,
                            generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32,
                            generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.bfloat16)

    # Per-block FP8 quantize for weights
    w1_fp8, w1_scale = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, w2_scale = _per_block_quantize_fp8(w2_bf16)

    # Input
    hidden = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1

    # Quantize input to FP8
    hidden_fp8 = hidden.to(torch.float8_e4m3fn)
    # Input scale: [K//128, total_tokens] — transposed block scale
    k_blocks = (K + 127) // 128
    hidden_scale = torch.ones(k_blocks, total_tokens, device=device, dtype=torch.float32)

    # Routing logits for desired distribution
    routing_logits = torch.full((total_tokens, E), -10.0, device=device, dtype=torch.bfloat16)
    offset = 0
    for expert_id in range(E):
        for _ in range(tokens_per_expert[expert_id]):
            if offset < total_tokens:
                routing_logits[offset, expert_id] = 10.0
                offset += 1

    def run():
        return trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_fp8,
            hidden_states_scale=hidden_scale,
            gemm1_weights=w1_fp8,
            gemm1_weights_scale=w1_scale,
            gemm2_weights=w2_fp8,
            gemm2_weights_scale=w2_scale,
            num_experts=E, top_k=top_k,
            n_group=None, topk_group=None,
            intermediate_size=N,
            local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,  # Renormalize: TopK -> Softmax
            do_finalize=True,
            tune_max_num_tokens=max(total_tokens, 4096),
        )

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 7. CUTLASS FP8 Per-Tensor (via CutlassExpertsFp8)
# =========================================================================

def _bench_cutlass_fp8_per_tensor(E, tokens_per_expert, K, N):
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import CutlassExpertsFp8
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertForwardPayload
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig

    device = "cuda"
    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E
    total_tokens = sum(tokens_per_expert)
    top_k = min(8, E)

    w1 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32,
                       generator=torch.Generator(device).manual_seed(SEED)) * 0.1).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, K, N, device=device, dtype=torch.float32,
                       generator=torch.Generator(device).manual_seed(SEED + 1)) * 0.1).to(torch.float8_e4m3fn)
    w1_scale = torch.ones(E, dtype=torch.float32, device=device)
    w2_scale = torch.ones(E, dtype=torch.float32, device=device)

    weights = {W.moe_w1: w1, W.moe_w2: w2, W.moe_s1: w1_scale, W.moe_s2: w2_scale}

    config = _make_config(E, K, N, top_k, max(tokens_per_expert))
    quant_config = FusedMoEQuantConfig(
        quant_dtype=torch.float8_e4m3fn, per_act_token_quant=True, per_out_ch_quant=False, block_shape=None)
    executor = CutlassExpertsFp8(config, quant_config, weights)

    hidden = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16,
                          generator=torch.Generator(device).manual_seed(SEED + 2)) * 0.1

    # Construct topk_ids for desired distribution
    topk_ids = torch.zeros(total_tokens, top_k, device=device, dtype=torch.int32)
    topk_weights = torch.ones(total_tokens, top_k, device=device, dtype=torch.bfloat16) / top_k
    offset = 0
    for expert_id in range(E):
        for _ in range(tokens_per_expert[expert_id]):
            if offset < total_tokens:
                topk_ids[offset, 0] = expert_id
                offset += 1

    payload = ExpertForwardPayload(
        expert_x=hidden, expert_x_origin_dtype=torch.bfloat16,
        expert_x_scale=None,
        expert_topk_ids=topk_ids, expert_topk_weights=topk_weights)

    def run():
        return executor.execute(payload, "SiGLU", None, None, False, None)

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None


# =========================================================================
# 8. CUTLASS FP4 Standalone (vLLM/SGLang kernel) — placeholder
# =========================================================================

def _bench_cutlass_fp4_standalone(E, tokens_per_expert, K, N):
    """Placeholder for vLLM/SGLang CUTLASS FP4 GemmUniversal.
    Requires standalone compilation of nvfp4_blockwise_moe_kernel.cu.
    See cutlass_fp4_standalone.py for the compilation wrapper.
    """
    try:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.cutlass_fp4_standalone import bench_cutlass_fp4
        return bench_cutlass_fp4(E, tokens_per_expert, K, N, SEED, WARMUP_ITERS, BENCH_ITERS)
    except ImportError:
        return None, None, "cutlass_fp4_standalone not available (requires JIT compilation)"
    except Exception as e:
        return None, None, str(e)[:200]


# =========================================================================
# Config helper
# =========================================================================

def _make_config(E, K, N, top_k, max_batch):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import ParallelismConfig, MoeConfig
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter

    mc = ModelConfig()
    mc.attn_config.head_num = 2; mc.attn_config.size_per_head = 128
    mc.num_layers = 2; mc.max_seq_len = 2048; mc.vocab_size = 500000
    mc.expert_num = E; mc.hidden_size = K; mc.moe_inter_size = N; mc.moe_k = top_k
    pc = ParallelismConfig()
    pc.world_size = 1; pc.dp_size = 1; pc.tp_size = 1; pc.ep_size = 1
    pc.dp_rank = 0; pc.tp_rank = 0; pc.ep_rank = 0; pc.world_rank = 0
    pc.local_rank = 0; pc.local_world_size = 1
    moe_cfg = MoeConfig(); moe_cfg.ll_num_max_token = max_batch
    return MoEConfigAdapter(model_config=mc, parallelism_config=pc, moe_config=moe_cfg)


# =========================================================================
# Scenarios
# =========================================================================

# (label, E, tokens_per_expert, N, K, stype)
# tokens_per_expert: int for uniform, list[int] for non-uniform

# Per-expert M values: 8, 16, 64, 256, 512, 1024, 2048
UNIFORM_SCENARIOS = [
    ("M/E=8",       8,     8, 2048, 7168, "decode"),
    ("M/E=16",      8,    16, 2048, 7168, "decode"),
    ("M/E=64",      8,    64, 2048, 7168, "prefill"),
    ("M/E=256",     8,   256, 2048, 7168, "prefill"),
    ("M/E=512",     8,   512, 2048, 7168, "prefill"),
    ("M/E=1024",    8,  1024, 2048, 7168, "prefill"),
    ("M/E=2048",    8,  2048, 2048, 7168, "prefill"),
    ("E64-M/E=8",  64,     8, 2048, 7168, "decode"),
]


def _make_skewed_a(total_m, E=8):
    """Long-tail: 1 hot expert gets ~38%, rest follow geometric decay."""
    ratios = [0.375, 0.25, 0.125, 0.094, 0.063, 0.047, 0.031, 0.015]
    counts = [max(1, int(r * total_m)) for r in ratios[:E]]
    diff = total_m - sum(counts)
    counts[0] += diff
    return counts


def _make_skewed_b(total_m, E=8):
    """Extreme: 1 expert gets ~75%, rest get scraps."""
    counts = [1] * E
    counts[0] = total_m - (E - 1)
    return counts


# Non-uniform activation for each M/E level
# total_M = uniform M/E * E, then redistribute non-uniformly
SKEWED_SCENARIOS = []
for m_per_e in [8, 64, 256, 512, 1024, 2048]:
    total_m = m_per_e * 8
    sk_a = _make_skewed_a(total_m)
    sk_b = _make_skewed_b(total_m)
    SKEWED_SCENARIOS.append(
        (f"SkA-totM={total_m}", 8, sk_a, 2048, 7168, "skewed"))
    SKEWED_SCENARIOS.append(
        (f"SkB-totM={total_m}", 8, sk_b, 2048, 7168, "skewed"))

ALL_SCENARIOS = UNIFORM_SCENARIOS + SKEWED_SCENARIOS

# Implementation registry: (short_name, bench_fn, is_fused_moe)
IMPLEMENTATIONS = [
    ("FP4-CD",      _bench_cutedsl_fp4,          False),
    ("FP4-TRT",     _bench_trtllm_fp4,           True),
    ("FP4-CUT",     _bench_cutlass_fp4_standalone, False),
    ("FP8-FI",      _bench_flashinfer_fp8,       False),
    ("FP8-DGM",     _bench_deepgemm_masked,      False),
    ("FP8-DGC",     _bench_deepgemm_contiguous,  False),
    ("FP8-CUT",     _bench_cutlass_fp8_per_tensor, True),
    ("FP8-TRT",     _bench_trtllm_fp8,           True),
]


class TestUnifiedMoeBenchmark(unittest.TestCase):

    def test_all_implementations(self):
        """Unified Full MoE benchmark: 3 FP4 + 5 FP8 implementations."""
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        impl_names = [name for name, _, _ in IMPLEMENTATIONS]
        col_width = 12

        print("\n" + "=" * 170)
        print(f"  Unified MoE Benchmark — SM100 (Full MoE: FC1+SiLU+FC2)")
        print(f"  N=2048 (inter), K=7168 (hidden), Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}, Seed={SEED}")
        print(f"  Implementations: {', '.join(impl_names)}")
        print(f"  * = fused end-to-end MoE (includes routing+gather overhead)")
        print("=" * 170)

        # Print header
        hdr = f"{'Scenario':<22} {'Type':<8} {'E':>4} {'TotM':>7} {'maxM/E':>6} | "
        for name, _, is_fused in IMPLEMENTATIONS:
            mark = "*" if is_fused else ""
            hdr += f"{name + mark:>{col_width}} {'TF':>7} | "
        hdr += f"{'Best':>12}"
        print(hdr)
        print("-" * 170)

        all_errors = []

        for label, E, tpe, N, K, stype in ALL_SCENARIOS:
            if isinstance(tpe, int):
                total_tokens = E * tpe
                max_m_per_e = tpe
            else:
                total_tokens = sum(tpe)
                max_m_per_e = max(tpe)

            results = {}
            for name, bench_fn, _ in IMPLEMENTATIONS:
                ms, tf, err = _safe_run(lambda fn=bench_fn: fn(E, tpe, K, N))
                results[name] = (ms, tf, err)
                if err:
                    all_errors.append(f"[{name}] {label}: {err}")

            # Format output
            row = f"{label:<22} {stype:<8} {E:>4} {total_tokens:>7} {max_m_per_e:>6} | "
            candidates = []
            for name, _, is_fused in IMPLEMENTATIONS:
                ms, tf, _ = results[name]
                ms_s = f"{ms:.3f}" if ms else "ERR"
                tf_s = f"{tf:.1f}" if tf else "-"
                row += f"{ms_s:>{col_width}} {tf_s:>7} | "
                if ms:
                    candidates.append((name, ms))

            best = min(candidates, key=lambda x: x[1])[0] if candidates else "N/A"
            row += f"{best:>12}"
            print(row)

        print("=" * 170)
        print("Legend: FP4-CD=CuteDSL, FP4-TRT=TRT-LLM FP4, FP4-CUT=CUTLASS FP4 (vLLM/SGLang),")
        print("        FP8-FI=FlashInfer groupwise, FP8-DGM=DeepGEMM masked, FP8-DGC=DeepGEMM contiguous,")
        print("        FP8-CUT=CUTLASS per-tensor, FP8-TRT=TRT-LLM FP8")
        print("  * = fused end-to-end MoE (includes routing + gather overhead)")
        print(f"  Workload: Full MoE (FC1[M,K]x[2N,K]^T + SiLU + FC2[M,N]x[K,N]^T)")
        print(f"  SkA = long-tail (38%/25%/12.5%/...), SkB = extreme (75%/rest=1 each)")

        if _AUTOTUNE_LOG:
            print(f"\nAutotune Details ({len(_AUTOTUNE_LOG)}):")
            for entry in _AUTOTUNE_LOG:
                print(f"  {entry}")

        if all_errors:
            print(f"\nErrors ({len(all_errors)}):")
            for e in all_errors:
                print(f"  {e}")


if __name__ == "__main__":
    unittest.main()
