"""V4 decode-arm FP8 sparse attention op.

Wraps FlashMLA's ``flash_mla_with_kvcache(is_fp8_kvcache=True)`` for
single- and dual-pool decode. The kernel reads the packed
``fp8_model1_mla`` KV cache directly (no dequant on the read path) and
outputs bf16 attention output.

Dual-pool support uses FlashMLA's ``extra_k_cache`` +
``extra_indices_in_kvcache`` parameters to attend over a second FP8 pool
(CSA / HCA compressor pool) in a single kernel call, with in-kernel
softmax merging across both pools (mirrors vLLM
``deepseek_v4_attention.py:849-865``). Replaces the legacy "dequant both
pools -> BF16 cat -> TileLang sparse_attn" path which was
bandwidth-bound on the dequant kernels.

FlashMLA wheel is required (CUDA >= 12.9). The op asserts wheel
availability at forward — there is no slow Python reference fallback
because all dev/CI/prod boxes carry flash_mla.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

_REFERENCE_ENV = "RTP_LLM_FLASH_MLA_DECODE_REFERENCE"

_FLASH_MLA_AVAILABLE = False
try:
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 9):
            from flash_mla import (
                flash_mla_with_kvcache,  # type: ignore[import-not-found]
            )
            from flash_mla import get_mla_metadata  # type: ignore[import-not-found]

            _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    logging.warning("[dsv4-fp8] flash_mla wheel unavailable (%s)", e)


def _use_reference_decode(q: torch.Tensor) -> bool:
    forced = os.environ.get(_REFERENCE_ENV, "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if not q.is_cuda:
        return True
    try:
        major, _ = torch.cuda.get_device_capability(q.device)
    except Exception:
        return True
    if major < 10:
        return True
    try:
        import flash_mla  # noqa: F401
    except Exception:
        return True
    return False


def _as_3d_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() == 4:
        return indices.squeeze(2).contiguous()
    return indices.contiguous()


def _dequantize_v4_pool_slots(pool_3d: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """584-byte MODEL1 fallback for a V4 (not V4.1) pool."""
    from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_kv_quant_decode_op import (
        dequantize_v4_kv_slot,
        read_model1_kv_slot_bytes,
    )

    rows = slots.reshape(-1)
    out = torch.zeros(
        (rows.numel(), 512), dtype=torch.bfloat16, device=pool_3d.device
    )
    block_size = int(pool_3d.shape[1])
    for i, slot in enumerate(rows.tolist()):
        if slot < 0:
            continue
        raw = read_model1_kv_slot_bytes(
            pool_3d, slot // block_size, slot % block_size, block_size
        )
        out[i] = dequantize_v4_kv_slot(raw)
    return out


def _dequantize_pool_slots(
    pool_3d: torch.Tensor, slots: torch.Tensor, *, global_fp4: bool
) -> torch.Tensor:
    width = int(pool_3d.shape[-1])
    if global_fp4:
        if width == 288:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
                dequantize_k_cache_slots_fp4,
            )

            return dequantize_k_cache_slots_fp4(pool_3d, slots)
        if width == 584:
            return _dequantize_v4_pool_slots(pool_3d, slots)
        raise ValueError(f"unsupported global KV pool width {width}")
    if width == 528:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            dequantize_swa_k_cache,
        )

        return dequantize_swa_k_cache(pool_3d, slots)
    if width == 584:
        return _dequantize_v4_pool_slots(pool_3d, slots)
    raise ValueError(f"unsupported SWA KV pool width {width}")


_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _pool_blocks(pool_3d: torch.Tensor) -> torch.Tensor:
    stride = int(pool_3d.stride(0))
    return pool_3d.as_strided((pool_3d.shape[0], stride), (stride, 1))


def _decode_pool_keys(
    pool_3d: torch.Tensor, slots: torch.Tensor, *, global_fp4: bool
) -> torch.Tensor:
    """Graph-safe gather+dequant for one decode step.

    All operations have static shapes: invalid slots are clamped to slot 0 and
    then zeroed arithmetically, so CUDA graph capture sees no host sync and no
    data-dependent shape.
    """
    width = int(pool_3d.shape[-1])
    if width == 584:
        return _dequantize_v4_pool_slots(pool_3d, slots)
    blocks = _pool_blocks(pool_3d)
    block_size = int(pool_3d.shape[1])
    valid = slots >= 0
    safe = slots.clamp_min(0)
    block = safe // block_size
    pos = safe % block_size
    rows = blocks[block]
    if global_fp4:
        if width != 288:
            raise ValueError(f"unsupported global KV pool width {width}")
        cols = torch.arange(288, device=pool_3d.device)
        raw = torch.gather(rows, 1, pos[:, None] * 288 + cols[None, :])
        payload = raw[:, :256]
        scales = raw[:, 256:288].contiguous().view(torch.float8_e4m3fn)
        codes = torch.stack((payload & 0x0F, payload >> 4), dim=-1).reshape(
            raw.shape[0], 512
        )
        table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=raw.device)
        values = table[codes.to(torch.int64)]
        values = values * scales.to(torch.float32).repeat_interleave(16, dim=1)
    else:
        if width != 528:
            raise ValueError(f"unsupported SWA KV pool width {width}")
        cols = torch.arange(512, device=pool_3d.device)
        payload = torch.gather(rows, 1, pos[:, None] * 528 + cols[None, :])
        scols = torch.arange(16, device=pool_3d.device)
        scales = torch.gather(
            rows, 1, pos[:, None] * 528 + 512 + scols[None, :]
        )
        values = payload.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        values = values.reshape(-1, 16, 32) * torch.exp2(
            scales.to(torch.float32) - 127.0
        )[:, :, None]
        values = values.reshape(-1, 512)
    return values * valid.unsqueeze(-1)


class SparseAttnV4DecodeFp8Op:
    """FP8 sparse attention decode op (single- or dual-pool).

    Args (forward):
      q          : ``[B, q_len, n_heads, head_dim]`` bf16
      kv_cache   : ``[num_blocks, block_size, entry_bytes]`` uint8 packed FP8
        primary pool (584B V4 or 528B V4.1 SWA in dual-pool mode).
      attn_sink  : ``[n_heads]`` fp32 — per-head learned sink
      topk_idxs  : ``[B, q_len, topk]`` int32 — per-request global slot
        ids into the primary pool.
      cache_seqlens : unused in sparse FP8 decode; accepted to keep the
        attention helper call sites uniform.
      block_table   : unused in sparse FP8 decode; FlashMLA consumes global
        slot ids from ``topk_idxs`` directly.
      topk_length        : optional ``[B]`` int32 — per-request leftmost
        valid length on ``topk_idxs``.
      extra_k_cache      : optional secondary packed pool (584B V4 FP8 or
        288B V4.1 FP4). Triggers FlashMLA's dual-pool path.
      extra_topk_idxs    : optional ``[B, q_len, extra_topk]`` int32 —
        global slot ids into ``extra_k_cache``.
      extra_topk_length  : optional ``[B]`` int32 — per-request leftmost
        valid length on ``extra_topk_idxs``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        softmax_scale: float,
    ) -> None:
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        """Single- or dual-pool sparse attention.

        ``sched_meta`` is the FlashMLA planner output (``get_mla_metadata``
        return) — owned by :class:`DSv4DecodeAttnMetadataFP8` and fetched via
        :func:`~decode_attn_metadata.get_or_build_sched_meta`. The op is a
        pure dispatcher; it does NOT cache sched_meta itself (was the
        iter2 anti-pattern that accumulated per-instance state across
        decode steps).

        Dual-pool: pass ``extra_k_cache`` (e.g. CMP pool 3D
        ``[num_blocks, block_size, 584]`` uint8) + ``extra_topk_idxs``
        (3D ``[B, q_len, extra_topk]`` int32 global slot ids) to attend
        over a second FP8 KV pool in a single FlashMLA invocation. The
        kernel merges softmax across both pools natively.

        ``return_lse=True`` additionally returns FlashMLA's softmax LSE
        ``[B, num_heads_q, seq_len_q]`` fp32 (the raw ``log Z`` — unaffected
        by ``attn_sink``). Empty rows report ``+inf`` and must be guarded
        by the consumer.
        """
        if q.ndim == 3:
            # Warmup/graph and the SM80 reference Q path may hand us the
            # [B, H, D] form with an implicit q_len=1.  The native sparse
            # MLA ABI and the reference fallback both require [B, S, H, D].
            q = q.unsqueeze(1)
        if _use_reference_decode(q):
            return self._forward_reference(
                q,
                kv_cache,
                attn_sink,
                topk_idxs,
                topk_length,
                extra_k_cache,
                extra_topk_idxs,
                extra_topk_length,
                return_lse,
            )
        return self._forward_flash_mla(
            q,
            kv_cache,
            attn_sink,
            topk_idxs,
            sched_meta,
            cache_seqlens,
            block_table,
            topk_length,
            extra_k_cache,
            extra_topk_idxs,
            extra_topk_length,
            return_lse,
        )

    def _forward_reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        topk_length: Optional[torch.Tensor],
        extra_k_cache: Optional[torch.Tensor],
        extra_topk_idxs: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        return_lse: bool,
    ) -> Any:
        """Gather the selected FP8/FP4 KV rows and run MLA in torch.

        Pre-SM100 devices have no FlashMLA sparse decode kernel.  Decoding
        only reads ``topk_length`` slots per request, so materializing those
        rows is bounded and does not create a persistent BF16 KV mirror.
        Both pools are merged with the same log-sum-exp as the native
        dual-pool kernel, including the per-head attention sink.
        """
        batch, q_len, heads, _ = q.shape
        primary = _as_3d_indices(topk_idxs)
        extra = _as_3d_indices(extra_topk_idxs) if extra_topk_idxs is not None else None
        sink = attn_sink.reshape(-1).to(torch.float32) if attn_sink is not None else None
        out = torch.zeros(
            (batch, q_len, heads, self.head_dim), dtype=q.dtype, device=q.device
        )
        lse = torch.full(
            (batch, heads, q_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            length = None if topk_length is None else topk_length.reshape(-1)[b]
            extra_length = (
                None
                if extra_topk_length is None
                else extra_topk_length.reshape(-1)[b]
            )
            for t in range(q_len):
                slots = primary[b, t]
                valid = slots >= 0
                if length is not None:
                    valid = valid & (
                        torch.arange(slots.numel(), device=q.device) < length
                    )
                keys = _decode_pool_keys(kv_cache, slots, global_fp4=False).float()
                if extra is not None:
                    eslots = extra[b, t]
                    evalid = eslots >= 0
                    if extra_length is not None:
                        evalid = evalid & (
                            torch.arange(eslots.numel(), device=q.device) < extra_length
                        )
                    ekeys = _decode_pool_keys(
                        extra_k_cache, eslots, global_fp4=True
                    ).float()
                    keys = torch.cat((keys, ekeys), dim=0)
                    valid = torch.cat((valid, evalid), dim=0)
                query = q[b, t].to(torch.float32)
                logits = query @ keys.transpose(0, 1) * self.softmax_scale
                logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
                max_logit = logits.amax(dim=-1).clamp_min(-1e30)
                if sink is not None:
                    max_all = torch.maximum(max_logit, sink)
                    denom = torch.exp(logits - max_all.unsqueeze(-1)).sum(-1)
                    denom = denom + torch.exp(sink - max_all)
                else:
                    max_all = max_logit
                    denom = torch.exp(logits - max_all.unsqueeze(-1)).sum(-1)
                acc = torch.exp(logits - max_all.unsqueeze(-1)) @ keys
                out[b, t] = torch.where(
                    denom.unsqueeze(-1) > 0, acc / denom.clamp_min(1e-30).unsqueeze(-1), 0
                ).to(q.dtype)
                lsum = torch.exp(logits - max_all.unsqueeze(-1)).sum(-1)
                lse[b, :, t] = torch.where(
                    lsum > 0,
                    max_all + torch.log(lsum.clamp_min(1e-30)),
                    torch.full_like(max_all, float("-inf")),
                )
        if return_lse:
            return out, lse
        return out

    def _forward_flash_mla(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv41.flash_mla import (
            flash_mla_with_kvcache,  # type: ignore[import-not-found]
        )

        B, q_len, H, D = q.shape
        # FlashMLA expects 4D q ``(batch_size, seq_len_q, num_heads_q, head_dim)``
        # and 3D indices ``(batch_size, seq_len_q, topk)`` per the installed
        # wheel's ``flash_mla_interface.flash_mla_with_kvcache`` docstring.

        # FlashMLA FP8 kernel requires 4D k_cache: [num_blocks, block_size, num_heads_k=1, kv_dim].
        kv_4d = kv_cache.unsqueeze(-2)
        extra_kv_4d = extra_k_cache.unsqueeze(-2) if extra_k_cache is not None else None

        # topk_idxs: [B, q_len, topk] preferred; collapse a stray num_heads_k axis if present.
        if topk_idxs.dim() == 4:
            topk_3d = topk_idxs.squeeze(2).contiguous()
        else:
            topk_3d = topk_idxs.contiguous()

        if extra_topk_idxs is not None:
            extra_topk_3d = (
                extra_topk_idxs.squeeze(2).contiguous()
                if extra_topk_idxs.dim() == 4
                else extra_topk_idxs.contiguous()
            )
        else:
            extra_topk_3d = None

        # Sparse FlashMLA consumes global slot ids from ``indices`` directly.
        # Its sparse branch does not pass block_table/cache_seqlens to the CUDA
        # kernel, so keep dense metadata disabled here.
        block_table = None
        cache_seqlens = None

        # DSv4 attn_sink is per-head fp32, loaded from ckpt (layers.*.attn.attn_sink
        # shape [n_heads], non-zero ~0.3..0.6 mean). FlashMLA kernel applies
        # output *= exp(lse) / (exp(lse) + exp(attn_sink)). Mirrors vLLM
        # ``deepseek_v4_attention.py:860`` (both single- and dual-pool calls).
        attn_out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_4d,
            block_table=block_table,
            head_dim_v=self.head_dim,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            is_fp8_kvcache=True,
            indices=topk_3d,
            softmax_scale=self.softmax_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_kv_4d,
            extra_indices_in_kvcache=extra_topk_3d,
            extra_topk_length=extra_topk_length,
        )

        attn_out = attn_out.view(B, q_len, H, self.head_dim).contiguous()
        if return_lse:
            return attn_out, lse
        return attn_out
