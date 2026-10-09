"""Scoped reference-recipe replacements for checkpoint accuracy diagnosis only."""

import os
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import attention as base_attention
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.decode import compute_qkv as decode_attention
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb


def swa_roundtrip(values):
    shape, dtype = values.shape, values.dtype
    groups = values.float().reshape(-1, 16, 32)
    scales = (groups.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
    quantized = (groups / scales[..., None]).to(torch.float8_e4m3fn).float()
    return (quantized * scales[..., None]).reshape(shape).to(dtype)


def rounded_norm_rope(
    values,
    weight,
    freqs,
    rope_head_dim,
    *,
    eps=1e-6,
    inverse=False,
    out=None,
    inplace=False,
    group_heads=None,
    round_norm_to_input_dtype=True,
):
    assert out is None or not inplace
    normalized = values.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + eps
    )
    if weight is not None:
        normalized = normalized * weight.float()
    normalized = normalized.to(values.dtype)
    if rope_head_dim:
        frequencies = freqs.reshape(-1, rope_head_dim // 2)
        tail = (
            normalized[..., -rope_head_dim:]
            .float()
            .reshape(len(frequencies), -1, rope_head_dim // 2, 2)
        )
        rotated = (
            torch.view_as_complex(tail)
            * (frequencies.conj() if inverse else frequencies)[:, None, :]
        )
        normalized[..., -rope_head_dim:] = torch.view_as_real(rotated).reshape(
            *values.shape[:-1], rope_head_dim
        )
    target = values if inplace else out
    if target is not None:
        target.copy_(normalized)
        return target
    return normalized


def bf16_output_projection(layer, output, freqs, out=None):
    output = output.reshape(-1, layer.n_heads, layer.head_dim)
    apply_rotary_emb(
        output[..., -layer.rope_head_dim :].unsqueeze(0), freqs, inverse=True
    )
    grouped = output.reshape(len(output), layer.n_groups, -1)
    assert layer.wo_a_s.dtype == torch.float8_e8m0fnu
    # Keep only this layer's dequantized weight. Retaining all 40 would add
    # several GiB per worker. Match the reference's grouped contraction shape.
    scale = layer.wo_a_s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    weight = (layer.wo_a_w.float() * scale).bfloat16()
    weight = weight.view(layer.n_groups, layer.o_lora_rank, -1)
    projected = torch.einsum("tgd,grd->tgr", grouped, weight)
    return layer.wo_b(projected.flatten(1), out=out)


@contextmanager
def reference_precision(profile):
    if profile not in ("baseline", "router_fp32", "attention", "all"):
        raise ValueError(f"Unknown checkpoint precision diagnostic: {profile}")
    with ExitStack() as stack:
        if profile in ("router_fp32", "all"):
            stack.enter_context(patch.dict(os.environ, {"DSV4_GATE_FP32": "1"}))
        if profile in ("attention", "all"):
            original = AttentionV41FP8._ensure_prefill_kv_full

            def quantized_fresh(layer, qkv, common):
                qkv = original(layer, qkv, common)
                return qkv._replace(kv_full=swa_roundtrip(qkv.kv_full))

            stack.enter_context(
                patch.object(
                    AttentionV41FP8, "_ensure_prefill_kv_full", quantized_fresh
                )
            )
            stack.enter_context(
                patch.object(AttentionV41FP8, "_project_output", bf16_output_projection)
            )
            stack.enter_context(
                patch.object(base_attention, "fused_rmsnorm_rope", rounded_norm_rope)
            )
            stack.enter_context(
                patch.object(decode_attention, "fused_rmsnorm_rope", rounded_norm_rope)
            )
        yield
