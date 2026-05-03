"""DSv4 IEEE-precise RMSNorm — Python wrapper around the vendored
flashinfer kernel patched with `__frcp_rn(__fsqrt_rn(...))`.

============================================================================
Why this exists (root cause writeup, 2026-05-03)
============================================================================
The framework C++ ``rtp_llm_ops.rmsnorm`` (commit 19496bdf5 perf-replaced
the torch fp32 chain with it) routes through flashinfer's
``norm.cuh::RMSNorm`` -> ``math::rsqrt`` (~ 2 ULP, FTZ approx PTX
``rsqrt.approx.ftz.f32``).  At S=64K + DSv4 60-layer residual + tiny
``attn_norm.weight`` (~0.03), this ~2 ULP per-layer error accumulates
into a directional bias of ~0.5 logits on a specific token (id=271 for
the smoke prompt) — flipping argmax to an adjacent candidate (id=223)
and producing context-leakage gibberish ("Thepackage..." fragments of
the prompt rather than coherent prose).

Bisect data, captured by the lm_head logits dump comparing revert (good)
vs re-applied 19496bdf5 (bad) at S=65599 (saved as /tmp/dsv4_logits/
{good,bad}.pt + compare.py):

  GOOD (revert: torch fp32 chain, fp32 weight)
    rank 0: id 271  logit 10.5512
    rank 1: id 223  logit 10.5399    gap 0.011 to top1
  BAD (re-applied: flashinfer C++ rmsnorm, bf16 weight)
    rank 0: id 223  logit 10.4196
    rank 1: id  16  logit 10.0067
    rank 2: id 271  logit  9.9203    -- good's top1 dropped 0.63 logit

cosine(good, bad) logits = 0.987 (whole-vector very close), but the
specific-token shift on 271 was systematic, not random noise.  Confirmed
by colleague's screenshot showing the 1-line patch of flashinfer's
``math::rsqrt`` to ``__frcp_rn(__fsqrt_rn(...))`` flips the output back
to coherent.

This module wires the patched kernel from
``rtp_llm/models_py/bindings/cuda/kernels/dsv4_ieee_rmsnorm.{cu,h}`` (built
into ``librtp_compute_ops.so``) into the dsv4 forward path, replacing the
broken ``rtp_llm_ops.rmsnorm`` calls in block.py / compressor.py /
attention._rmsnorm_weighted.

The Q/K per-head RMSNorm + partial RoPE site (``fused_rmsnorm_rope`` in
``modules/dsv4/_fused_rmsnorm_rope_triton.py``) had the same root cause
on the Triton side (``tl.rsqrt`` lowers to ``rsqrt.approx.f32``); patched
in-place to ``1.0/tl.sqrt`` (which lowers to IEEE ``sqrt.rn + rcp.rn``,
verified 0 ULP in /tmp/test_rsqrt_variants.py).
"""

from __future__ import annotations

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """In-place RMSNorm.  Same signature as ``flashinfer.norm.rmsnorm``.

    Args:
        x:      ``[N, D]`` bf16 contiguous.
        weight: ``[D]``    bf16 contiguous.
        out:    ``[N, D]`` bf16 contiguous (may alias ``x``).
        eps:    Variance epsilon.
    """
    stream_id = torch.cuda.current_stream().cuda_stream
    rtp_llm_ops.dsv4_ieee_rmsnorm(out, x, weight, eps, stream_id)
