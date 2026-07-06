"""Wrapper around vllm-xpu-kernels ops for use in rtp-llm.

Provides optimized SYCL/DPC++ kernels for Intel XPU via vllm-xpu-kernels.
Falls back to PyTorch native ops when vllm-xpu-kernels is not available.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_VLLM_XPU_AVAILABLE = False
_FA2_AVAILABLE = False
_MOE_AVAILABLE = False

# If vllm-xpu-kernels is pip-installed, import works directly.
# Only use VLLM_XPU_KERNELS_PATH for development / editable installs.
_vllm_xpu_root = os.environ.get("VLLM_XPU_KERNELS_PATH", "")
if _vllm_xpu_root and os.path.isdir(_vllm_xpu_root) and _vllm_xpu_root not in sys.path:
    sys.path.insert(0, _vllm_xpu_root)

try:
    import vllm_xpu_kernels._C  # noqa: F401
    _VLLM_XPU_AVAILABLE = True
    logger.info("vllm-xpu-kernels _C loaded")
except (ImportError, OSError, RuntimeError) as exc:
    logger.warning("vllm-xpu-kernels _C not available: %s", exc)

try:
    import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401
    _FA2_AVAILABLE = True
    logger.info("vllm-xpu-kernels FA2 loaded")
except (ImportError, OSError, RuntimeError) as exc:
    logger.warning("vllm-xpu-kernels FA2 not available: %s", exc)

try:
    import vllm_xpu_kernels._moe_C  # noqa: F401
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    _MOE_AVAILABLE = True
    logger.info("vllm-xpu-kernels MoE loaded")
except (ImportError, OSError, RuntimeError) as exc:
    logger.warning("vllm-xpu-kernels MoE not available: %s", exc)


def is_available():
    return _VLLM_XPU_AVAILABLE

def is_fa2_available():
    return _FA2_AVAILABLE

def is_moe_available():
    return _MOE_AVAILABLE


def rms_norm(result, input, weight, epsilon):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.rms_norm(result, input, weight, epsilon)
    else:
        variance = input.pow(2).mean(-1, keepdim=True)
        normed = input * torch.rsqrt(variance + epsilon)
        result.copy_(normed * weight)


def fused_add_rms_norm(input, residual, weight, epsilon):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
    else:
        residual.add_(input)
        variance = residual.pow(2).mean(-1, keepdim=True)
        normed = residual * torch.rsqrt(variance + epsilon)
        input.copy_(normed * weight)


def silu_and_mul(out, input):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.silu_and_mul(out, input)
    else:
        d = input.shape[-1] // 2
        x, gate = input[..., :d], input[..., d:]
        out.copy_(torch.nn.functional.silu(x) * gate)


def gelu_and_mul(out, input):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.gelu_and_mul(out, input)
    else:
        d = input.shape[-1] // 2
        x, gate = input[..., :d], input[..., d:]
        out.copy_(torch.nn.functional.gelu(x) * gate)


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
    else:
        # PyTorch fallback: apply rotary embeddings using cos/sin cache
        rotary_dim = cos_sin_cache.shape[-1]
        half_dim = rotary_dim // 2
        cos_sin = cos_sin_cache[positions.long()]
        cos = cos_sin[:, :half_dim]
        sin = cos_sin[:, half_dim:]
        # Apply to query
        _apply_rotary_inplace(query, cos, sin, head_size, half_dim, is_neox)
        # Apply to key
        _apply_rotary_inplace(key, cos, sin, head_size, half_dim, is_neox)


def _apply_rotary_inplace(x, cos, sin, head_size, half_dim, is_neox):
    """Apply rotary embedding in-place to a [num_tokens, num_heads * head_size] tensor."""
    num_tokens = x.shape[0]
    total = x.shape[-1]
    num_heads = total // head_size
    x_view = x.view(num_tokens, num_heads, head_size)
    if is_neox:
        x1 = x_view[..., :half_dim]
        x2 = x_view[..., half_dim:2*half_dim]
        cos_e = cos.unsqueeze(1)
        sin_e = sin.unsqueeze(1)
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x2 * cos_e + x1 * sin_e
        x_view[..., :half_dim] = o1
        x_view[..., half_dim:2*half_dim] = o2
    else:
        # Interleaved style
        x1 = x_view[..., 0:2*half_dim:2]
        x2 = x_view[..., 1:2*half_dim:2]
        cos_e = cos.unsqueeze(1)
        sin_e = sin.unsqueeze(1)
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x2 * cos_e + x1 * sin_e
        x_view[..., 0:2*half_dim:2] = o1
        x_view[..., 1:2*half_dim:2] = o2


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                      max_seqlen_q, max_seqlen_k,
                      softmax_scale=None, causal=True,
                      block_table=None, seqused_k=None):
    if _FA2_AVAILABLE:
        from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
        # block_table requires seqused_k and forbids cu_seqlens_k
        if block_table is not None:
            return flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                causal=causal,
                block_table=block_table,
            )
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
        )
    else:
        if block_table is not None:
            raise RuntimeError(
                "flash_attn_varlen SDPA fallback does not support block_table (paged KV cache). "
                "Install vllm-xpu-kernels to enable paged attention on XPU."
            )
        if seqused_k is not None:
            raise RuntimeError(
                "flash_attn_varlen SDPA fallback does not support seqused_k. "
                "Install vllm-xpu-kernels to enable this feature on XPU.")
        return _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                     max_seqlen_q, max_seqlen_k, softmax_scale, causal)


def _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, softmax_scale, causal):
    assert cu_seqlens_q is not None and cu_seqlens_k is not None, \
        "_sdpa_varlen_fallback requires cu_seqlens_q/cu_seqlens_k"
    import torch.nn.functional as F
    # Copy cu_seqlens to CPU once to avoid per-request D2H sync in the loop.
    cu_q_cpu = cu_seqlens_q.cpu() if cu_seqlens_q.is_cuda or (hasattr(cu_seqlens_q, 'is_xpu') and cu_seqlens_q.is_xpu) else cu_seqlens_q
    cu_k_cpu = cu_seqlens_k.cpu() if cu_seqlens_k.is_cuda or (hasattr(cu_seqlens_k, 'is_xpu') and cu_seqlens_k.is_xpu) else cu_seqlens_k
    batch_size = cu_q_cpu.numel() - 1
    outputs = []
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    for i in range(batch_size):
        q_start, q_end = cu_q_cpu[i].item(), cu_q_cpu[i + 1].item()
        k_start, k_end = cu_k_cpu[i].item(), cu_k_cpu[i + 1].item()
        qi = q[q_start:q_end].unsqueeze(0).transpose(1, 2)
        ki = k[k_start:k_end].unsqueeze(0).transpose(1, 2)
        vi = v[k_start:k_end].unsqueeze(0).transpose(1, 2)
        # GQA/MQA: repeat K/V heads to match Q head count for SDPA.
        if qi.shape[1] != ki.shape[1]:
            n_rep = qi.shape[1] // ki.shape[1]
            ki = ki.repeat_interleave(n_rep, dim=1)
            vi = vi.repeat_interleave(n_rep, dim=1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale)
        outputs.append(oi.transpose(1, 2).squeeze(0))
    if outputs:
        return torch.cat(outputs, dim=0)
    return q.new_empty(0, q.shape[1], q.shape[2])
