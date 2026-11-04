"""FlashMLA-backed sparse MLA path (optional, auto-detected).

``flash_mla.flash_mla_sparse_fwd`` (container package ``flash-mla``) computes the
same absorbed-MQA sparse attention over selected latent slots as
``sparse_mla_triton.sparse_mla_fwd`` but is ~1.9x faster on B300 (baseline-log
#5.133) at the cost of two API quirks:

* ``h_q`` must be 64 or 128 -> we zero-pad our 32 heads to 64 and slice back;
* the kernel returns ``(output_bf16, max_logits, lse)``.

Numerical equivalence was verified against our kernel on 128k-context cases:
out diff 3.7e-05 (bf16 output rounding), lse diff 9.5e-07 (baseline-log #5.134).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["flashmla_available", "sparse_flashmla_fwd"]


def flashmla_available() -> bool:
    """True when the FlashMLA sparse kernel can actually run on this device.

    The kernel is compiled for SM90a/SM100f(+); importing the package is not
    enough (it installs everywhere), so the compute capability is checked too.
    """
    try:
        import flash_mla  # noqa: F401

        if not hasattr(flash_mla, "flash_mla_sparse_fwd"):
            return False
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


def sparse_flashmla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    *,
    kv_lora_rank: int = 512,
    out_dtype: Optional[torch.dtype] = None,
    scale: Optional[float] = None,
    **triton_kwargs: object,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA via FlashMLA, same contract as ``sparse_mla_fwd``.

    Args:
        q: ``[T, H, DQ]`` absorbed queries (H must be <= 64; padded to 64).
        kv: ``[N, DQ]`` latent KV rows (``[c_kv ; k_rope]``), DQ = kv_lora_rank + rope.
        slots: ``[T, S]`` int indices into ``kv``; ``-1`` marks padding.
        kv_lora_rank: latent rank (512 for LongCat/DeepSeek MLA).
        out_dtype: output dtype (defaults to ``q.dtype``); the kernel returns bf16.
        scale: softmax scale (defaults to ``1 / sqrt(DQ)``).

    Returns:
        ``(out [T, H, kv_lora_rank], lse [T, H])`` with lse in natural log.

    Extra keyword arguments accepted by the Triton kernels (``block_n``,
    ``num_warps``, ``num_stages``, ``kv_scales``, ...) are ignored so this
    wrapper is a drop-in replacement.
    """
    del triton_kwargs  # Triton-only tuning knobs do not apply to FlashMLA
    from flash_mla import flash_mla_sparse_fwd

    t, h, dq = q.shape
    if h > 64:
        raise ValueError(f"sparse_flashmla_fwd supports at most 64 heads, got {h}")
    rope = dq - kv_lora_rank
    if scale is None:
        scale = 1.0 / (dq ** 0.5)

    q_pad = q
    if h != 64:
        pad = torch.zeros((t, 64 - h, dq), dtype=q.dtype, device=q.device)
        q_pad = torch.cat([q, pad], dim=1).contiguous()

    kv3 = kv if kv.dim() == 3 else kv.unsqueeze(1)          # [N, 1, DQ]
    idx = slots.to(torch.int32)
    if idx.dim() == 3:                                       # already [T, 1, S]
        idx = idx.squeeze(1)
    # FlashMLA requires the slot-list length to be a multiple of 128 (1008 is
    # rejected by the SM100 sparse kernel); pad with -1, which the kernel treats
    # as invalid (baseline-log #5.136).
    s_len = idx.shape[1]
    if s_len % 128:
        pad_to = (s_len + 127) // 128 * 128
        tail = torch.full((idx.shape[0], pad_to - s_len), -1, dtype=idx.dtype, device=idx.device)
        idx = torch.cat([idx, tail], dim=1)
    idx3 = idx.unsqueeze(1)                                  # [T, 1, S_pad]

    out, _max_logits, lse = flash_mla_sparse_fwd(
        q_pad, kv3, idx3, sm_scale=float(scale), d_v=kv_lora_rank,
    )
    want = out_dtype or q.dtype
    return out[:, :h].to(want), lse[:, :h].float()
