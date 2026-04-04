"""
Benchmark: Prefill Attention Backends on H20 (SM90)

Backends:
1. Dao-AILab FA3 (flash_attn_interface.flash_attn_varlen_func)
2. Dao-AILab FA2 (flash_attn.flash_attn_varlen_func)
3. FlashInfer fa3 ragged (BatchPrefillWithRaggedKVCacheWrapper backend="fa3")
4. FlashInfer fa2 ragged (BatchPrefillWithRaggedKVCacheWrapper backend="fa2")
5. FlashInfer fa3 paged (BatchPrefillWithPagedKVCacheWrapper backend="fa3")
6. FlashInfer fa2 paged (BatchPrefillWithPagedKVCacheWrapper backend="fa2")
7. TRT FMHA V2 (TRTAttnOp C++ pybind op, pure prefill only)
8. PyTorch SDPA - reference

Usage:
    pytest bench_prefill_attention_backends.py -v -s --remote-session --remote-gpu-type=H20
"""

import importlib.util as _ilu
import logging
import os as _os
import statistics
from typing import Dict

import pytest
import torch

_bench_utils_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "bench_utils.py"
)
_spec = _ilu.spec_from_file_location("_bench_utils", _bench_utils_path)
_bench_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bench_utils)
attention_tflops_per_sec = _bench_utils.attention_tflops_per_sec
bench_gpu_time = _bench_utils.bench_gpu_time_with_cupti
set_seed = _bench_utils.set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Data preparation
# ============================================================================


def _make_qkv(batch_size, q_len, kv_len, num_heads, kv_heads, head_dim, dtype, device):
    total_q = batch_size * q_len
    total_kv = batch_size * kv_len
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv, kv_heads, head_dim, dtype=dtype, device=device)
    cu_q = torch.arange(
        0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
    )
    cu_k = torch.arange(
        0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device=device
    )
    return q, k, v, cu_q, cu_k


def _fill_paged_kv(
    k, v, batch_size, kv_len, kv_heads, head_dim, page_size, dtype, device
):
    """Create paged KV cache filled with k/v data."""
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    kv_data = torch.zeros(
        total_pages, 2, kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_seq
    )
    for b in range(batch_size):
        s = b * kv_len
        for p in range(pages_per_seq):
            pg_start = p * page_size
            pg_end = min(pg_start + page_size, kv_len)
            pg_len = pg_end - pg_start
            page_idx = block_tables[b, p].item()
            kv_data[page_idx, 0, :, :pg_len, :] = k[
                s + pg_start : s + pg_end
            ].transpose(0, 1)
            kv_data[page_idx, 1, :, :pg_len, :] = v[
                s + pg_start : s + pg_end
            ].transpose(0, 1)
    return kv_data, block_tables


# ============================================================================
# Backend wrappers
# ============================================================================


def run_fa3_native(q, k, v, cu_q, cu_k, max_q, max_k):
    from flash_attn_interface import flash_attn_varlen_func

    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
    return out[0] if isinstance(out, tuple) else out


def run_fa2_native(q, k, v, cu_q, cu_k, max_q, max_k):
    from flash_attn import flash_attn_varlen_func

    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
    return out[0] if isinstance(out, tuple) else out


def run_sdpa_reference(q, k, v, cu_q, cu_k, num_heads, kv_heads, head_dim):
    batch_size = len(cu_q) - 1
    outputs = []
    for i in range(batch_size):
        sq, eq = cu_q[i].item(), cu_q[i + 1].item()
        sk, ek = cu_k[i].item(), cu_k[i + 1].item()
        q_len_i = eq - sq
        kv_len_i = ek - sk
        qi = q[sq:eq].transpose(0, 1).unsqueeze(0)
        ki = k[sk:ek].transpose(0, 1).unsqueeze(0)
        vi = v[sk:ek].transpose(0, 1).unsqueeze(0)
        if kv_heads != num_heads:
            ki = ki.repeat_interleave(num_heads // kv_heads, dim=1)
            vi = vi.repeat_interleave(num_heads // kv_heads, dim=1)
        if q_len_i == kv_len_i:
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True
            )
        else:
            attn_mask = torch.zeros(q_len_i, kv_len_i, dtype=qi.dtype, device=qi.device)
            for r in range(q_len_i):
                allowed = kv_len_i - q_len_i + r + 1
                if allowed < kv_len_i:
                    attn_mask[r, allowed:] = float("-inf")
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=attn_mask
            )
        outputs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _try_create_trt_op(num_heads, kv_heads, head_dim, dtype, batch_size, q_len, kv_len):
    """Try to create TRT V2 FMHA op. Returns (op, params, packed_qkv_fn) or None."""
    if q_len != kv_len:
        return None
    try:
        from rtp_llm.ops import AttentionConfigs
        from rtp_llm.ops.compute_ops import PyAttentionInputs, TRTAttnOp, get_typemeta

        ac = AttentionConfigs()
        ac.head_num = num_heads
        ac.kv_head_num = kv_heads
        ac.size_per_head = head_dim
        ac.is_causal = True
        ac.tokens_per_block = 64
        ac.kernel_tokens_per_block = 64
        ac.max_seq_len = q_len

        device = "cuda"
        inp = PyAttentionInputs()
        inp.is_prefill = True
        inp.dtype = get_typemeta(torch.zeros(1, dtype=dtype))
        inp.input_lengths = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device=device
        )
        inp.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        inp.sequence_lengths = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device=device
        )
        inp.cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
        )
        inp.cu_kv_seqlens = inp.cu_seqlens.clone()
        inp.context_total_kv_length = batch_size * kv_len
        inp.total_tokens = batch_size * q_len
        inp.is_s_padded = False

        op = TRTAttnOp(ac)
        if not op.support(inp):
            logger.warning("TRT-V2: support() returned False")
            return None
        params = op.prepare(inp)
        return op, params
    except Exception as e:
        import traceback

        logger.warning(f"TRT-V2 init failed: {e}\n{traceback.format_exc()}")
        return None


def _try_create_trt_paged_op(
    num_heads,
    kv_heads,
    head_dim,
    dtype,
    batch_size,
    q_len,
    kv_len,
    q,
    k,
    v,
    page_size=64,
):
    """Try to create TRT V2 Paged FMHA op.

    Data flow (matching framework TRTPagedMHAImpl.forward):
    1. RoPE+KVCache write puts ALL K/V (prefix + new) into paged kv_cache
    2. TRT Paged kernel reads packed QKV (new tokens only) + paged kv_cache (prefix K/V)
    3. Kernel internally merges prefix K/V from cache + new K/V from packed input

    Returns (op, params, kv_cache, packed_qkv) or None.
    """
    if q_len == kv_len:
        return None  # Paged TRT needs prefix (q_len < kv_len)
    try:
        import math

        from rtp_llm.ops import AttentionConfigs
        from rtp_llm.ops.compute_ops import (
            LayerKVCache,
            PyAttentionInputs,
            TRTPagedAttnOp,
            get_typemeta,
        )

        device = "cuda"
        prefix_len = kv_len - q_len

        ac = AttentionConfigs()
        ac.head_num = num_heads
        ac.kv_head_num = kv_heads
        ac.size_per_head = head_dim
        ac.is_causal = True
        ac.tokens_per_block = page_size
        ac.kernel_tokens_per_block = page_size
        ac.max_seq_len = kv_len

        # Block table covers ALL kv_len pages (prefix + new token pages)
        pages_per_seq = math.ceil(kv_len / page_size)
        total_pages = batch_size * pages_per_seq
        kv_cache_block_id = torch.zeros((batch_size, pages_per_seq), dtype=torch.int32)
        for b in range(batch_size):
            kv_cache_block_id[b, :pages_per_seq] = torch.arange(
                b * pages_per_seq, (b + 1) * pages_per_seq, dtype=torch.int32
            )

        inp = PyAttentionInputs()
        inp.is_prefill = True
        inp.dtype = get_typemeta(torch.zeros(1, dtype=dtype))
        inp.input_lengths = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device=device
        )
        inp.prefix_lengths = torch.full(
            (batch_size,), prefix_len, dtype=torch.int32, device=device
        )
        inp.sequence_lengths = torch.full(
            (batch_size,), kv_len, dtype=torch.int32, device=device
        )
        inp.cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
        )
        inp.cu_kv_seqlens = torch.arange(
            0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device=device
        )
        inp.context_total_kv_length = batch_size * kv_len
        inp.total_tokens = batch_size * q_len
        inp.is_s_padded = False
        inp.kv_cache_kernel_block_id_host = kv_cache_block_id
        inp.kv_cache_kernel_block_id_device = kv_cache_block_id.to(device)
        inp.kv_cache_block_id_host = kv_cache_block_id
        inp.kv_cache_block_id_device = kv_cache_block_id.to(device)

        op = TRTPagedAttnOp(ac)
        if not op.support(inp):
            logger.warning("TRT-V2-paged: support() returned False")
            return None
        params = op.prepare(inp)

        # Build paged KV cache with ONLY prefix K/V (prefix_len tokens per batch)
        # New token K/V comes from packed QKV input, written by RoPE op in framework
        kv_cache_tensor = torch.zeros(
            total_pages, 2, kv_heads, page_size, head_dim, dtype=dtype, device=device
        )
        for b in range(batch_size):
            kv_start = b * kv_len  # offset into flat k/v arrays
            for p in range(pages_per_seq):
                pg_s = p * page_size
                pg_e = min(pg_s + page_size, prefix_len)  # only fill prefix portion
                if pg_s >= prefix_len:
                    break
                pg_l = pg_e - pg_s
                pi = b * pages_per_seq + p
                kv_cache_tensor[pi, 0, :, :pg_l, :] = k[
                    kv_start + pg_s : kv_start + pg_e
                ].transpose(0, 1)
                kv_cache_tensor[pi, 1, :, :pg_l, :] = v[
                    kv_start + pg_s : kv_start + pg_e
                ].transpose(0, 1)

        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_tensor

        # Packed QKV for new tokens: [total_q_tokens, (H + 2*KVH) * D]
        # Q from q tensor, K/V from the new-token portion of k/v tensors
        total_q = batch_size * q_len
        qkv_dim = (num_heads + 2 * kv_heads) * head_dim
        packed = torch.empty(total_q, qkv_dim, dtype=dtype, device=device)
        for b in range(batch_size):
            qs = b * q_len
            # new token K/V starts at offset prefix_len within this batch's kv segment
            ks = b * kv_len + prefix_len
            packed[qs : qs + q_len, : num_heads * head_dim] = q[
                qs : qs + q_len
            ].reshape(q_len, num_heads * head_dim)
            packed[
                qs : qs + q_len,
                num_heads * head_dim : (num_heads + kv_heads) * head_dim,
            ] = k[ks : ks + q_len].reshape(q_len, kv_heads * head_dim)
            packed[qs : qs + q_len, (num_heads + kv_heads) * head_dim :] = v[
                ks : ks + q_len
            ].reshape(q_len, kv_heads * head_dim)

        return op, params, kv_cache, packed
    except Exception as e:
        import traceback

        logger.warning(f"TRT-V2-paged init failed: {e}\n{traceback.format_exc()}")
        return None


# ============================================================================
# Benchmark helper
# ============================================================================


def _bench_backend(
    name,
    fn,
    ref_output,
    q_len,
    kv_len,
    batch_size,
    head_dim,
    num_heads,
    warmup_iters,
    repeat_iters,
    results,
):
    try:
        out = fn()
        max_diff = (out.reshape(ref_output.shape) - ref_output).abs().max().item()
        times = bench_gpu_time(
            fn, dry_run_iters=warmup_iters, repeat_iters=repeat_iters
        )
        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        tflops = attention_tflops_per_sec(
            batch_size, q_len, kv_len, head_dim, head_dim, num_heads, True, mean_ms
        )
        results[name] = {
            "ms": mean_ms,
            "std_ms": std_ms,
            "tflops": tflops,
            "max_diff": max_diff,
        }
    except Exception as e:
        logger.warning(f"{name} skipped: {e}")


# ============================================================================
# Benchmark runner
# ============================================================================


def benchmark_one_config(
    batch_size: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup_iters: int = 3,
    repeat_iters: int = 10,
) -> Dict[str, Dict[str, float]]:
    device = "cuda"
    set_seed(42)
    q, k, v, cu_q, cu_k = _make_qkv(
        batch_size, q_len, kv_len, num_heads, kv_heads, head_dim, dtype, device
    )
    results = {}
    common = dict(
        q_len=q_len,
        kv_len=kv_len,
        batch_size=batch_size,
        head_dim=head_dim,
        num_heads=num_heads,
        warmup_iters=warmup_iters,
        repeat_iters=repeat_iters,
        results=results,
    )

    # Reference
    ref_output = run_sdpa_reference(q, k, v, cu_q, cu_k, num_heads, kv_heads, head_dim)

    # 1. Dao-AILab FA3
    _bench_backend(
        "DaoAILab-FA3",
        lambda: run_fa3_native(q, k, v, cu_q, cu_k, q_len, kv_len),
        ref_output,
        **common,
    )

    # 2. Dao-AILab FA2
    _bench_backend(
        "DaoAILab-FA2",
        lambda: run_fa2_native(q, k, v, cu_q, cu_k, q_len, kv_len),
        ref_output,
        **common,
    )

    # 3. FlashInfer fa3 ragged
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = BatchPrefillWithRaggedKVCacheWrapper(
            float_workspace_buffer=ws, kv_layout="NHD", backend="fa3"
        )
        w.plan(
            cu_q,
            cu_k,
            num_heads,
            kv_heads,
            head_dim,
            head_dim,
            causal=True,
            q_data_type=dtype,
        )
        _bench_backend("FI-fa3-ragged", lambda: w.run(q, k, v), ref_output, **common)
    except Exception as e:
        logger.warning(f"FI-fa3-ragged skipped: {e}")

    # 4. FlashInfer fa2 ragged
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = BatchPrefillWithRaggedKVCacheWrapper(
            float_workspace_buffer=ws, kv_layout="NHD", backend="fa2"
        )
        w.plan(
            cu_q,
            cu_k,
            num_heads,
            kv_heads,
            head_dim,
            head_dim,
            causal=True,
            q_data_type=dtype,
        )
        _bench_backend("FI-fa2-ragged", lambda: w.run(q, k, v), ref_output, **common)
    except Exception as e:
        logger.warning(f"FI-fa2-ragged skipped: {e}")

    # 5/6. FlashInfer paged (fa3 + fa2) — HND layout [num_pages, 2, kv_heads, page_size, head_dim]
    for paged_backend, paged_name in [("fa3", "FI-fa3-paged"), ("fa2", "FI-fa2-paged")]:
        try:
            from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

            page_size = 64
            pages_per_seq = (kv_len + page_size - 1) // page_size
            total_pages = batch_size * pages_per_seq
            # KV cache: [total_pages, 2, kv_heads, page_size, head_dim] (HND)
            kv_paged = torch.zeros(
                total_pages,
                2,
                kv_heads,
                page_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for b in range(batch_size):
                s = b * kv_len
                for p in range(pages_per_seq):
                    pg_s = p * page_size
                    pg_e = min(pg_s + page_size, kv_len)
                    pg_l = pg_e - pg_s
                    pi = b * pages_per_seq + p
                    kv_paged[pi, 0, :, :pg_l, :] = k[s + pg_s : s + pg_e].transpose(
                        0, 1
                    )  # [kv_heads, pg_l, head_dim]
                    kv_paged[pi, 1, :, :pg_l, :] = v[s + pg_s : s + pg_e].transpose(
                        0, 1
                    )

            paged_kv_indptr = torch.arange(
                0,
                (batch_size + 1) * pages_per_seq,
                pages_per_seq,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_indices = torch.arange(
                total_pages, dtype=torch.int32, device=device
            )
            last_page_len = kv_len - (pages_per_seq - 1) * page_size
            paged_kv_last_page_len = torch.full(
                (batch_size,), last_page_len, dtype=torch.int32, device=device
            )

            ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            pw = BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=ws, kv_layout="HND", backend=paged_backend
            )
            pw.plan(
                cu_q,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_heads,
                kv_heads,
                head_dim,
                page_size,
                causal=True,
                q_data_type=dtype,
            )
            _bench_backend(
                paged_name,
                lambda pw=pw, q=q, kv=kv_paged: pw.run(q, kv),
                ref_output,
                **common,
            )
        except Exception as e:
            import traceback

            logger.warning(f"{paged_name} skipped: {e}\n{traceback.format_exc()}")

    # 7. TRT FMHA V2 (pure prefill only)
    trt_result = _try_create_trt_op(
        num_heads, kv_heads, head_dim, dtype, batch_size, q_len, kv_len
    )
    if trt_result is not None:
        trt_op, trt_params = trt_result
        total = batch_size * q_len
        qkv_dim = (num_heads + 2 * kv_heads) * head_dim
        packed = torch.empty(total, qkv_dim, dtype=dtype, device=device)
        packed[:, : num_heads * head_dim] = q.reshape(total, num_heads * head_dim)
        packed[:, num_heads * head_dim : (num_heads + kv_heads) * head_dim] = k.reshape(
            total, kv_heads * head_dim
        )
        packed[:, (num_heads + kv_heads) * head_dim :] = v.reshape(
            total, kv_heads * head_dim
        )
        _bench_backend(
            "TRT-V2",
            lambda: trt_op.forward(packed, None, trt_params),
            ref_output,
            **common,
        )

    # 8. TRT FMHA V2 Paged (prefill-with-cache only, q_len < kv_len)
    trt_paged_result = _try_create_trt_paged_op(
        num_heads, kv_heads, head_dim, dtype, batch_size, q_len, kv_len, q, k, v
    )
    if trt_paged_result is not None:
        trt_p_op, trt_p_params, trt_p_kvcache, trt_p_packed = trt_paged_result
        _bench_backend(
            "TRT-V2-paged",
            lambda: trt_p_op.forward(trt_p_packed, trt_p_kvcache, trt_p_params),
            ref_output,
            **common,
        )

    # 9. SDPA reference timing
    _bench_backend(
        "SDPA-ref",
        lambda: run_sdpa_reference(q, k, v, cu_q, cu_k, num_heads, kv_heads, head_dim),
        ref_output,
        **common,
    )
    if "SDPA-ref" in results:
        results["SDPA-ref"]["max_diff"] = 0.0

    return results


# ============================================================================
# Validation
# ============================================================================

H100_FA3_FA2_SPEEDUP_RANGE = (1.3, 2.5)
H20_FP16_PEAK_TFLOPS = 296.0


def validate_results(all_results):
    ok = True
    for config_key, results in all_results.items():
        fa3 = results.get("DaoAILab-FA3")
        fa2 = results.get("FI-fa2-ragged") or results.get("DaoAILab-FA2")
        if fa3 and fa2 and fa2["tflops"] > 0:
            ratio = fa3["tflops"] / fa2["tflops"]
            lo, hi = H100_FA3_FA2_SPEEDUP_RANGE
            status = "OK" if lo <= ratio <= hi else "WARN"
            logger.info(
                f"  [{status}] {config_key}: FA3/FA2 = {ratio:.2f}x (expected {lo}-{hi}x)"
            )
            if status == "WARN":
                ok = False
        if fa3:
            util = fa3["tflops"] / H20_FP16_PEAK_TFLOPS * 100
            logger.info(
                f"  FA3 util: {util:.1f}% of {H20_FP16_PEAK_TFLOPS} TFLOPS peak"
            )
    return ok


# ============================================================================
# Test configs
# ============================================================================

BENCH_CONFIGS = [
    # (batch_size, q_len, kv_len, num_heads, kv_heads, head_dim, dtype_name)
    # --- Scenario A: Pure prefill (q_len == kv_len), GQA 32/8 ---
    (1, 1024, 1024, 32, 8, 128, "fp16"),
    (2, 1024, 1024, 32, 8, 128, "fp16"),
    (1, 4096, 4096, 32, 8, 128, "fp16"),
    (2, 4096, 4096, 32, 8, 128, "fp16"),
    (1, 16384, 16384, 32, 8, 128, "fp16"),
    (2, 16384, 16384, 32, 8, 128, "fp16"),
    (1, 32768, 32768, 32, 8, 128, "fp16"),
    # --- Scenario B: Prefill with existing KV cache ---
    # kv_cache=1k + new_prefill=1k/2k/4k
    (1, 1024, 2048, 32, 8, 128, "fp16"),
    (2, 1024, 2048, 32, 8, 128, "fp16"),
    (1, 2048, 3072, 32, 8, 128, "fp16"),
    (1, 4096, 5120, 32, 8, 128, "fp16"),
    # kv_cache=4k + new_prefill=1k/2k/4k
    (1, 1024, 5120, 32, 8, 128, "fp16"),
    (2, 1024, 5120, 32, 8, 128, "fp16"),
    (1, 2048, 6144, 32, 8, 128, "fp16"),
    (1, 4096, 8192, 32, 8, 128, "fp16"),
    (2, 4096, 8192, 32, 8, 128, "fp16"),
    # kv_cache=8k + new_prefill=1k/2k/4k
    (1, 1024, 9216, 32, 8, 128, "fp16"),
    (1, 2048, 10240, 32, 8, 128, "fp16"),
    (1, 4096, 12288, 32, 8, 128, "fp16"),
    # kv_cache=16k + new_prefill=1k/2k/4k
    (1, 1024, 17408, 32, 8, 128, "fp16"),
    (1, 2048, 18432, 32, 8, 128, "fp16"),
    (1, 4096, 20480, 32, 8, 128, "fp16"),
    # kv_cache=32k + new_prefill=1k/2k/4k
    (1, 1024, 33792, 32, 8, 128, "fp16"),
    (1, 2048, 34816, 32, 8, 128, "fp16"),
    (1, 4096, 36864, 32, 8, 128, "fp16"),
]


@pytest.mark.gpu(type="H20")
@pytest.mark.manual
@pytest.mark.timeout(600)
class TestBenchPrefillAttentionBackends:

    def test_benchmark_all(self):
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        all_results = {}

        header = f"{'Config':<45} {'Backend':<16} {'Mean(ms)':>10} {'Std(ms)':>10} {'TFLOPS':>8} {'MaxDiff':>10}"
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("=" * len(header))

        for bs, ql, kvl, nh, kvh, hd, dt in BENCH_CONFIGS:
            config_key = f"bs{bs}_q{ql}_kv{kvl}_h{nh}x{kvh}_d{hd}_{dt}"
            results = benchmark_one_config(bs, ql, kvl, nh, kvh, hd, dtype_map[dt])
            all_results[config_key] = results

            for backend, r in sorted(results.items()):
                logger.info(
                    f"  {config_key:<43} {backend:<16} {r['ms']:>8.3f}ms "
                    f"{r['std_ms']:>8.4f}ms {r['tflops']:>7.1f} {r['max_diff']:>9.5f}"
                )
                assert (
                    r["max_diff"] < 0.02
                ), f"{backend} max_diff={r['max_diff']:.5f} > 0.02 for {config_key}"

        logger.info("\n--- Validation against public baselines ---")
        validate_results(all_results)
