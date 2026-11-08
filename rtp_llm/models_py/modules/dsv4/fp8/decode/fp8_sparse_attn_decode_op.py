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

FlashMLA is used for non-SM120 devices; SM120 uses the FlashInfer adapter.
Provider loading is deferred until the execution device is bound; importing
the model does not load optional CUDA wheels.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
    get_sparse_attention_backend,
)

_FLASH_MLA_AVAILABLE = False
_flash_mla_load_attempted = False


def _load_flash_mla() -> bool:
    """Validate the optional FlashMLA wheel on first non-SM120 use.

    Importing the DSv4 model package is shared by CUDA variants that do not use
    FlashMLA.  Keep wheel loading out of module initialization so a missing or
    ABI-incompatible optional wheel cannot break those startup paths.
    """
    global _FLASH_MLA_AVAILABLE, _flash_mla_load_attempted
    if _FLASH_MLA_AVAILABLE:
        return True
    if _flash_mla_load_attempted:
        return False
    _flash_mla_load_attempted = True
    try:
        if not torch.version.cuda:
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) < (12, 9):
            return False
        flash_mla = importlib.import_module("rtp_llm.models_py.utils.flash_mla_legacy")
        getattr(flash_mla, "flash_mla_with_kvcache")
        getattr(flash_mla, "get_mla_metadata")
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as error:
        logging.info("[dsv4-fp8] FlashMLA unavailable (%s)", error)
        return False
    _FLASH_MLA_AVAILABLE = True
    return True


class SparseAttnV4DecodeFp8Op:
    """FP8 sparse attention decode op (single- or dual-pool).

    Args (forward):
      q          : ``[B, q_len, n_heads, head_dim]`` bf16
      kv_cache   : ``[num_blocks, block_size, 584]`` uint8 packed FP8
        primary pool (SWA in dual-pool mode).
      attn_sink  : ``[n_heads]`` fp32 — per-head learned sink
      topk_idxs  : ``[B, q_len, topk]`` int32 — per-request global slot
        ids into the primary pool.
      cache_seqlens : unused in sparse FP8 decode; accepted to keep the
        attention helper call sites uniform.
      block_table   : unused in sparse FP8 decode; FlashMLA consumes global
        slot ids from ``topk_idxs`` directly.
      topk_length        : optional ``[B]`` int32 — per-request leftmost
        valid length on ``topk_idxs``.
      extra_k_cache      : optional secondary FP8 pool (CMP). Triggers
        FlashMLA's dual-pool path.
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
        self._device: Optional[torch.device] = None
        self._forward_impl: Optional[Callable[..., torch.Tensor]] = None
        self._uses_flash_mla = True

    def bind_device(self, device: torch.device) -> None:
        """Prepare once for the actual query device, before metadata or execution.

        Construction can happen before weights reach their inference device.
        A bound op belongs to one device; moving the model requires a new op.
        """
        if device == self._device:
            return
        if self._device is not None:
            raise ValueError(
                f"sparse decode op is bound to {self._device}, got {device}"
            )
        if device.type == "cuda" and device.index is None:
            raise ValueError("sparse decode requires an explicit CUDA device index")
        forward_impl, uses_flash_mla = self._prepare_backend(device)
        # Publish the binding only after provider validation/warmup succeeds.
        self._forward_impl = forward_impl
        self._uses_flash_mla = uses_flash_mla
        self._device = device

    def _prepare_backend(
        self, device: torch.device
    ) -> tuple[Callable[..., torch.Tensor], bool]:
        backend = get_sparse_attention_backend(device)
        forward = backend.prepare_decode(device, self.n_heads, self.softmax_scale)
        if forward is not None:
            return forward, False
        if not _load_flash_mla():
            raise RuntimeError(
                "FlashMLA is required for non-SM120 FP8 sparse decode; "
                "install the cuda12_9/cuda13 RTP-LLM dependency set"
            )
        return self._forward_flash_mla, True

    def get_sched_meta(
        self,
        device: torch.device,
        attn_metadata: Any,
        *,
        batch_size: int,
        q_len: int,
        topk: int,
        extra_attn_type: Optional[str],
    ) -> Any:
        """Use the selected backend; keep planner state on step metadata."""
        self.bind_device(device)
        if not self._uses_flash_mla:
            return None
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )

        return get_or_build_sched_meta(
            attn_metadata,
            batch_size=batch_size,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=topk,
            extra_attn_type=extra_attn_type,
        )

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
        """
        self.bind_device(q.device)
        assert self._forward_impl is not None
        return self._forward_impl(
            q=q,
            kv_cache=kv_cache,
            attn_sink=attn_sink,
            topk_idxs=topk_idxs,
            sched_meta=sched_meta,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            topk_length=topk_length,
            extra_k_cache=extra_k_cache,
            extra_topk_idxs=extra_topk_idxs,
            extra_topk_length=extra_topk_length,
        )

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
    ) -> torch.Tensor:
        from rtp_llm.models_py.utils.flash_mla_legacy import (
            flash_mla_with_kvcache,  # type: ignore[import-not-found]
        )

        B, q_len, H, D = q.shape
        # FlashMLA expects 4D q ``(batch_size, seq_len_q, num_heads_q, head_dim)``
        # and 3D indices ``(batch_size, seq_len_q, topk)`` per the installed
        # wheel's ``flash_mla_interface.flash_mla_with_kvcache`` docstring.

        assert topk_idxs is not None, "FP8 sparse decode requires topk_idxs"

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
        attn_out, _ = flash_mla_with_kvcache(
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

        return attn_out.view(B, q_len, H, self.head_dim).contiguous()
