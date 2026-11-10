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

FlashMLA wheel is required (CUDA >= 12.9). The packaged sparse decode
kernel rejects pre-SM100 devices at launch ("only supported on SM100f
architectures"); there the op dispatches to an exact reference that
gather-dequantizes both pools through the branch's own 528B/288B codecs
and runs the same joint softmax with per-head sink and topk_length
semantics.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import torch

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


@lru_cache(maxsize=None)
def _flash_mla_native_supported(index) -> bool:
    """FlashMLA's sparse decode kernels ship for SM100 families only."""
    major, _ = torch.cuda.get_device_capability(index)
    return major >= 10


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
        if q.is_cuda and not _flash_mla_native_supported(q.device.index):
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
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        """Exact pre-SM100 decode: gather-dequant both pools, joint softmax.

        Mirrors the FlashMLA sparse decode contract on the flat query-row
        layout: per-request global slot ids with ``-1`` padding, per-request
        ``topk_length`` bounding the scanned prefix, one joint softmax across
        the SWA and (when present) compressed pools, and the per-head sink
        contributing ``exp(sink - max)`` to the denominator only. The packed
        528B E4M3/UE8M0 and 288B E2M1 rows decode exactly into bf16 through
        the branch's own codecs; scores accumulate in fp32. ``return_lse``
        yields the raw ``log Z`` without the sink term, and empty rows report
        ``+inf`` like the native kernel. All masks and lengths stay on device,
        so the path is CUDA-graph capture and replay safe.
        """
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            ENTRY_BYTES,
            dequantize_swa_k_cache,
        )

        B, q_len, H, D = q.shape
        if kv_cache.shape[-1] != ENTRY_BYTES:
            raise RuntimeError(
                "pre-SM100 sparse decode reference requires the V4.1 528B "
                f"SWA pool, got entry bytes {kv_cache.shape[-1]}"
            )
        rows = B * q_len
        if topk_idxs.dim() == 4:
            topk_idxs = topk_idxs.squeeze(2)
        swa_idxs = topk_idxs.reshape(rows, -1).to(torch.long)
        width = swa_idxs.shape[1]
        keys = [
            dequantize_swa_k_cache(kv_cache, swa_idxs.reshape(-1)).view(
                rows, width, 512
            )
        ]
        column = torch.arange(width, device=q.device)
        valid = swa_idxs >= 0
        if topk_length is not None:
            lengths = (
                topk_length.reshape(B, 1)
                .expand(B, q_len)
                .reshape(rows)
                .to(torch.long)
            )
            valid = valid & (column[None, :] < lengths[:, None])
        masks = [valid]
        if extra_k_cache is not None and extra_topk_idxs is not None:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
                FP4_GLOBAL_ENTRY_BYTES,
                dequantize_k_cache_slots_fp4,
            )

            if extra_k_cache.shape[-1] != FP4_GLOBAL_ENTRY_BYTES:
                raise RuntimeError(
                    "pre-SM100 sparse decode reference requires the V4.1 "
                    "288B FP4 extra pool, got entry bytes "
                    f"{extra_k_cache.shape[-1]}"
                )
            if extra_topk_idxs.dim() == 4:
                extra_topk_idxs = extra_topk_idxs.squeeze(2)
            extra_idxs = extra_topk_idxs.reshape(rows, -1).to(torch.long)
            extra_width = extra_idxs.shape[1]
            keys.append(
                dequantize_k_cache_slots_fp4(
                    extra_k_cache, extra_idxs.reshape(-1)
                ).view(rows, extra_width, 512)
            )
            extra_valid = extra_idxs >= 0
            if extra_topk_length is not None:
                extra_lengths = (
                    extra_topk_length.reshape(B, 1)
                    .expand(B, q_len)
                    .reshape(rows)
                    .to(torch.long)
                )
                extra_column = torch.arange(extra_width, device=q.device)
                extra_valid = extra_valid & (
                    extra_column[None, :] < extra_lengths[:, None]
                )
            masks.append(extra_valid)
        selected = torch.cat(keys, dim=1) if len(keys) > 1 else keys[0]
        mask = torch.cat(masks, dim=1) if len(masks) > 1 else masks[0]
        del keys, masks
        queries = q.reshape(rows, H, D).to(torch.float32)
        values = selected.to(torch.float32)
        del selected
        logits = torch.einsum("rhd,rkd->rhk", queries, values) * self.softmax_scale
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        scores_max = logits.amax(dim=-1).clamp_min(-1e30)
        exp_logits = torch.exp(logits - scores_max.unsqueeze(-1))
        sum_exp = exp_logits.sum(dim=-1)
        sink = attn_sink.reshape(H).to(torch.float32)
        total = sum_exp + torch.exp(sink[None, :] - scores_max)
        acc = torch.einsum("rhk,rkd->rhd", exp_logits, values)
        out = (acc / total.unsqueeze(-1)).to(q.dtype).view(B, q_len, H, self.head_dim)
        if return_lse:
            lse = torch.where(
                sum_exp > 0,
                scores_max + torch.log(sum_exp),
                torch.full_like(sum_exp, float("inf")),
            )
            return out, lse.view(B, q_len, H).permute(0, 2, 1).contiguous()
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
