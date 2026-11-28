"""RTP-LLM Native Paged-Cache Runtime & PD-Disaggregated Serving Support for Qwen 4.5-Flash.

Provides:
  1. PD CacheStore publication contract (``validate_qwen4_5_pd_cache_inputs``,
     ``prepare_qwen4_5_cache_store_writers``, ``publish_qwen4_5_layer_caches``)
     matching ``qwen4_exp_runtime/pd_publication.py`` & C++ ``WriteCacheStoreOp``.
  2. Complete 8-tag Paged ``KVCache`` bridge (``Qwen4_5PagedStateBridge``) mapping
     ``full``, ``linear``, ``gdn_consumer_q_conv``, ``full_attn_kv_sconv``,
     ``full_attn_output_sconv``, ``linear_attn_output_sconv``, ``ffn_output_sconv``,
     ``ple_conv_state``, and ``ple_ngram_ctx`` onto paged physical block pools,
     with full support for ragged multi-request batches, prefix reuse, and
     page-boundary transitions (``seq_len % tokens_per_block == 0``).
  3. ``Qwen4_5RTPModel(GptModelBase)`` — drop-in ``_py_model_class()`` entry point
     compatible with both compiled RTP-LLM ``ModelLoader`` / C++ PD scheduler and
     pure-Python standalone/NCCL two-process PD-disaggregated token generation.
  4. ABI-equivalent local ``LocalKVCache`` pool & ``PDDisaggregatedSession`` for
     immediate Prefill->CacheStore->Decode token generation verification.
"""

from __future__ import annotations

import array
import copy
import logging
import math
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from rtp_llm.config.qwen4_5_config import parse_qwen4_5_config_json
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
    FULL_ATTN_KV_SCONV_TAG,
    FULL_ATTN_KV_TAG,
    FULL_ATTN_OUTPUT_SCONV_TAG,
    FFN_OUTPUT_SCONV_TAG,
    GDN_CONSUMER_Q_CONV_TAG,
    GDN_CONV_STATE_TAG,
    GDN_SSM_STATE_TAG,
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    LINEAR_ATTN_OUTPUT_SCONV_TAG,
    LINEAR_ATTN_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    Qwen4_5KVCachePlan,
    build_qwen4_5_kv_cache_plan,
)
from rtp_llm.models_py.model_desc.qwen4_5 import (
    Qwen4_5DecoderLayer,
    Qwen4_5ForCausalLM,
    Qwen4_5Model,
)
import inspect
from rtp_llm.models_py.modules.qwen4_5.attention import (
    apply_attn_output_gate_flatten,
    dense_linear_bf16_or_tgv,
    gemma_rmsnorm_and_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import bind_qwen4_5_model_weights
from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
    build_qsa_neox_cos_sin,
    qsa_select_causal_from_projected,
    qsa_sparse_paged_prefill_attention,
    qsa_sparse_prefill_attention,
    score_and_select_qsa_paged,
    view_indexer_kv_pool,
    view_indexer_state_pool,
    write_qsa_decode_cache_gpu_,
    write_qsa_prefill_cache,
    write_qsa_verify_cache_gpu_,
)
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import (
    hc_sconv_decode_opaque_combine_norm_triton,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import (
    _triton_int_spec,
    causal_dwconv1d_decode_opaque_triton,
    causal_dwconv1d_decode_triton,
    causal_dwconv1d_ragged_prefill_with_snapshots_triton,
    causal_dwconv1d_verify_opaque_triton,
    causal_dwconv1d_verify_triton,
    triton_fast_launch,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    FullAttnKVCache,
    GDNLayerState,
    PLELayerState,
    Qwen4_5ArchConfig,
    Qwen4_5RuntimeState,
    SharedGDNInputs,
    ShortConvState,
)

# ---------------------------------------------------------------------------
# Optional compiled RTP-LLM bindings (graceful pure-Python fallback)
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - pure-Python / no-triton mode
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

try:
    from rtp_llm.models_py.model_desc.module_base import (
        GptModelBase as _RtpGptModelBase,
    )
    from rtp_llm.ops.compute_ops import (
        PyModelInputs as _RtpPyModelInputs,
        PyModelOutputs as _RtpPyModelOutputs,
    )

    _HAS_COMPILED_GPT_BASE = True
except Exception:  # pragma: no cover - pure-Python mode
    _RtpGptModelBase = nn.Module  # type: ignore[assignment,misc]
    _RtpPyModelInputs = None
    _RtpPyModelOutputs = None
    _HAS_COMPILED_GPT_BASE = False


_RUNTIME_TRITON_CACHE: dict = {}
_DTYPE_ITEMSIZE: Dict[torch.dtype, int] = {
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float8_e4m3fn: 1,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int16: 2,
    torch.float32: 4,
    torch.int32: 4,
    torch.float64: 8,
    torch.int64: 8,
}


def _dtype_itemsize(dtype: torch.dtype) -> int:
    sz = _DTYPE_ITEMSIZE.get(dtype)
    if sz is not None:
        return sz
    return int(torch.empty((), dtype=dtype).element_size())


QWEN4_5_CACHE_TAGS = frozenset(
    (
        FULL_ATTN_KV_TAG,
        LINEAR_ATTN_TAG,
        GDN_CONSUMER_Q_CONV_TAG,
        FULL_ATTN_KV_SCONV_TAG,
        FULL_ATTN_OUTPUT_SCONV_TAG,
        LINEAR_ATTN_OUTPUT_SCONV_TAG,
        FFN_OUTPUT_SCONV_TAG,
        PLE_STATE_TAG,
        PLE_NGRAM_CTX_TAG,
        INDEXER_KV_TAG,
        INDEXER_STATE_TAG,
    )
)

QWEN4_5_SIDE_TAGS = QWEN4_5_CACHE_TAGS - frozenset((FULL_ATTN_KV_TAG, LINEAR_ATTN_TAG))


# ---------------------------------------------------------------------------
# ABI-Equivalent Local Containers (100% field-compatible with compute_ops)
# ---------------------------------------------------------------------------


@dataclass
class LocalPyCacheStoreInputs:
    tag: str
    request_id: int = 0
    block_ids: Tuple[int, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalLayerKVCache:
    """Matches ``rtp_llm.ops.compute_ops.LayerKVCache``."""

    kv_cache_base: torch.Tensor
    seq_size_per_block: int
    layer_id: int = -1
    tag: str = FULL_ATTN_KV_TAG
    kv_scale_base: Optional[torch.Tensor] = None


class LocalKVCache:
    """Matches ``rtp_llm.ops.compute_ops.KVCache`` for all 8 Qwen 4.5-Flash tags."""

    def __init__(
        self,
        plan: Qwen4_5KVCachePlan,
        *,
        tokens_per_block: int = 16,
        num_blocks_per_tag: Optional[Dict[str, int]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: Optional[torch.dtype] = None,
        allocate_fp8_kv_scale: bool = False,
    ) -> None:
        self.plan = plan
        self.tokens_per_block = int(tokens_per_block)
        self.device = device
        self.dtype = dtype
        self.kv_dtype = kv_dtype if kv_dtype is not None else dtype
        self.allocate_fp8_kv_scale = bool(allocate_fp8_kv_scale)
        self.layer_caches: List[Dict[str, LocalLayerKVCache]] = [
            {} for _ in range(len(plan.layers))
        ]
        self._tags_present: List[str] = []
        default_blocks = 128

        cfg = plan.arch_config
        full_kv_dim = cfg.num_key_value_heads * cfg.head_dim
        k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        ssm_elems = (
            cfg.linear_num_value_heads
            * cfg.linear_key_head_dim
            * cfg.linear_value_head_dim
        )

        # Element packing for LINEAR pool: stored in float32 slots per block
        # [ssm_elems (fp32) | conv_bf16_packed_as_fp32] -> unified 4B slot view
        gdn_conv_elems = (2 * k_dim + v_dim) * max(0, cfg.linear_conv_kernel_dim - 1)
        self.linear_ssm_elems = ssm_elems
        self.linear_conv_elems = gdn_conv_elems
        self.linear_block_fp32_elems = ssm_elems + (gdn_conv_elems + 1) // 2

        for lp in plan.layers:
            lid = lp.layer_idx
            # 1. FullAttn Producer KV pool: [N, 2, B_tok, H_kv, D_h]
            if lp.layer_type == "full_attention" and lp.allocates_kv_cache:
                nb = (num_blocks_per_tag or {}).get(FULL_ATTN_KV_TAG, default_blocks)
                base = torch.zeros(
                    (nb, 2, self.tokens_per_block, lp.num_kv_heads, lp.head_dim),
                    dtype=self.kv_dtype,
                    device=device,
                )
                kv_scale_base = None
                if self.kv_dtype == torch.float8_e4m3fn and self.allocate_fp8_kv_scale:
                    kv_scale_base = torch.ones(
                        (nb, 2, lp.num_kv_heads, self.tokens_per_block),
                        dtype=torch.float32,
                        device=device,
                    )
                self._add_cache(lid, FULL_ATTN_KV_TAG, base, kv_scale_base=kv_scale_base)
                if lp.has_qsa_indexer_cache:
                    ratio = lp.indexer_compress_ratio
                    if self.tokens_per_block % ratio != 0:
                        raise ValueError(
                            f"LocalKVCache tokens_per_block ({self.tokens_per_block}) "
                            f"must be divisible by indexer_compress_ratio ({ratio})"
                        )
                    entries = self.tokens_per_block // ratio
                    nb_ikv = (num_blocks_per_tag or {}).get(INDEXER_KV_TAG, default_blocks)
                    ikv_base = torch.zeros(
                        (nb_ikv, entries, lp.indexer_head_dim * 2),
                        dtype=torch.uint8,
                        device=device,
                    )
                    self._add_cache(lid, INDEXER_KV_TAG, ikv_base)

                    nb_ist = (num_blocks_per_tag or {}).get(INDEXER_STATE_TAG, default_blocks)
                    ist_base = torch.zeros(
                        (nb_ist, 2 * ratio, lp.indexer_head_dim),
                        dtype=torch.float32,
                        device=device,
                    )
                    self._add_cache(lid, INDEXER_STATE_TAG, ist_base)

            # 2. GDN Producer/Normal LINEAR pool (SSM FP32 + conv BF16 packed as [N, fp32_slots])
            elif lp.layer_type == "linear_attention":
                if lp.allocates_ssm_state:
                    nb = (num_blocks_per_tag or {}).get(LINEAR_ATTN_TAG, default_blocks)
                    base = torch.zeros(
                        (nb, self.linear_block_fp32_elems),
                        dtype=torch.float32,
                        device=device,
                    )
                    self._add_cache(lid, LINEAR_ATTN_TAG, base)
                elif lp.has_gdn_conv:
                    nb = (num_blocks_per_tag or {}).get(
                        GDN_CONSUMER_Q_CONV_TAG, default_blocks
                    )
                    base = torch.zeros(
                        (nb, lp.gdn_conv_channels, lp.gdn_conv_state_len),
                        dtype=dtype,
                        device=device,
                    )
                    self._add_cache(lid, GDN_CONSUMER_Q_CONV_TAG, base)

            # 3. OPAQUE_STATE regions for this layer
            for reg in lp.opaque_regions:
                nb = (num_blocks_per_tag or {}).get(reg.tag, default_blocks)
                rdt = torch.int64 if reg.dtype == "I64" else (
                    torch.float32 if reg.dtype == "F32" else dtype
                )
                base = torch.zeros((nb, *reg.shape), dtype=rdt, device=device)
                self._add_cache(lid, reg.tag, base)

        # Track allocated physical block IDs per request slot (block 0 reserved null)
        self._next_free_block: Dict[str, int] = {t: 1 for t in self._tags_present}

    def _add_cache(
        self,
        layer_idx: int,
        tag: str,
        base: torch.Tensor,
        *,
        kv_scale_base: Optional[torch.Tensor] = None,
    ) -> None:
        self.layer_caches[layer_idx][tag] = LocalLayerKVCache(
            kv_cache_base=base,
            seq_size_per_block=self.tokens_per_block,
            layer_id=layer_idx,
            tag=tag,
            kv_scale_base=kv_scale_base,
        )
        if tag not in self._tags_present:
            self._tags_present.append(tag)

    def group_tags(self) -> List[str]:
        return list(self._tags_present)

    def layer_count(self) -> int:
        return len(self.layer_caches)

    def get_layer_cache(self, layer_idx: int, tag: str = FULL_ATTN_KV_TAG) -> LocalLayerKVCache:
        return self.layer_caches[layer_idx][tag]

    def get_layer_cache_groups(self, layer_idx: int) -> List[LocalLayerKVCache]:
        return list(self.layer_caches[layer_idx].values())

    def get_seq_size_per_block(self, tag: str) -> int:
        return self.tokens_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        return self.tokens_per_block

    def allocate_blocks(self, tag: str, num_blocks: int) -> List[int]:
        start = self._next_free_block[tag]
        first_base = next(
            c.kv_cache_base
            for layer_map in self.layer_caches
            if tag in layer_map
            for c in (layer_map[tag],)
        )
        cap = int(first_base.shape[0])
        if start + num_blocks > cap:
            raise RuntimeError(
                f"LocalKVCache pool {tag!r} exhausted: requested [{start}, {start + num_blocks}) >= {cap}"
            )
        self._next_free_block[tag] = start + num_blocks
        return list(range(start, start + num_blocks))


_PINNED_DMA_LOCAL = threading.local()


def _pin_stage_copy_to_device(cpu_flat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a 1D contiguous CPU tensor into a fresh CUDA tensor using a pooled pinned event ring."""
    n = int(cpu_flat.numel())
    if n == 0:
        return torch.empty((0,), dtype=cpu_flat.dtype, device=device)
    if device.type != "cuda":
        return cpu_flat.to(device=device)
    if torch.cuda.is_current_stream_capturing():
        return cpu_flat.to(device=device)
    elem_bytes = cpu_flat.element_size()
    need_bytes = n * elem_bytes
    dev_idx = int(device.index if device.index is not None else torch.cuda.current_device())
    rings: Dict[int, Tuple[List[Tuple[Optional[torch.Tensor], Optional[torch.cuda.Event]]], int]] = getattr(
        _PINNED_DMA_LOCAL, "rings", None
    )
    if rings is None:
        rings = {}
        _PINNED_DMA_LOCAL.rings = rings
    entry = rings.get(dev_idx)
    n_slots = 16
    if entry is None:
        slots: List[Tuple[Optional[torch.Tensor], Optional[torch.cuda.Event]]] = [(None, None) for _ in range(n_slots)]
        idx = 0
    else:
        slots, idx = entry
    buf_u8, ev = slots[idx]
    if ev is not None and not ev.query():
        found = -1
        for k in range(1, n_slots):
            cand = (idx + k) % n_slots
            c_ev = slots[cand][1]
            if c_ev is None or c_ev.query():
                found = cand
                break
        if found >= 0:
            idx = found
            buf_u8, ev = slots[idx]
        else:
            ev.synchronize()
    if buf_u8 is None or int(buf_u8.numel()) < need_bytes or (buf_u8.data_ptr() % max(4, elem_bytes)) != 0:
        cap = 4096
        while cap < need_bytes:
            cap *= 2
        try:
            buf_u8 = torch.empty((cap,), dtype=torch.uint8, device="cpu", pin_memory=True)
        except Exception:
            return cpu_flat.to(device=device)
    pin_typed = buf_u8[:need_bytes].view(cpu_flat.dtype)
    pin_typed.copy_(cpu_flat)
    dev_u8 = torch.empty((need_bytes,), dtype=torch.uint8, device=device)
    dev_typed = dev_u8.view(cpu_flat.dtype)
    dev_typed.copy_(pin_typed, non_blocking=True)
    new_ev = ev if ev is not None else torch.cuda.Event(enable_timing=False)
    new_ev.record(torch.cuda.current_stream(device))
    slots[idx] = (buf_u8, new_ev)
    rings[dev_idx] = (slots, (idx + 1) % n_slots)
    return dev_typed


def _copy_i32_list_to_dev(vals: Sequence[int], device: torch.device) -> torch.Tensor:
    n = len(vals)
    if n == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    arr = array.array("i", vals)
    cpu_t = torch.frombuffer(arr, dtype=torch.int32)
    return _pin_stage_copy_to_device(cpu_t, device)


@dataclass
class LocalPyAttentionInputs:
    """Field-compatible replica of ``PyAttentionInputs``."""

    is_prefill: bool
    input_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    kv_cache_block_id: torch.Tensor
    kv_cache_block_id_device: torch.Tensor
    kv_cache_kernel_block_id: Optional[torch.Tensor] = None
    kv_cache_kernel_block_id_device: Optional[torch.Tensor] = None
    combo_position_ids: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    is_s_padded: bool = False
    is_target_verify: bool = False
    is_draft_extend: bool = False
    cache_store_inputs: Optional[LocalPyCacheStoreInputs] = None
    cache_store_writer: Any = None
    total_tokens: int = 0
    _input_lengths_dev: Optional[torch.Tensor] = None
    _input_lengths_src_ver: int = -1
    _prefix_lengths_dev: Optional[torch.Tensor] = None
    _prefix_lengths_src_ver: int = -1
    _sequence_lengths_dev: Optional[torch.Tensor] = None

    def input_lengths_device(self) -> torch.Tensor:
        if self.input_lengths.device == self.kv_cache_block_id_device.device:
            return self.input_lengths
        if self._input_lengths_dev is not None and self._input_lengths_src_ver == self.input_lengths._version:
            return self._input_lengths_dev
        return _pin_stage_copy_to_device(self.input_lengths.contiguous().reshape(-1), self.kv_cache_block_id_device.device)

    def prefix_lengths_device(self) -> torch.Tensor:
        if self.prefix_lengths.device == self.kv_cache_block_id_device.device:
            return self.prefix_lengths
        if self._prefix_lengths_dev is not None and self._prefix_lengths_src_ver == self.prefix_lengths._version:
            return self._prefix_lengths_dev
        return _pin_stage_copy_to_device(self.prefix_lengths.contiguous().reshape(-1), self.kv_cache_block_id_device.device)


@dataclass
class LocalPyModelInputs:
    """Field-compatible replica of ``PyModelInputs``."""

    input_ids: torch.Tensor
    attention_inputs: Union[LocalPyAttentionInputs, Dict[str, LocalPyAttentionInputs]]
    combo_position_ids: Optional[torch.Tensor] = None
    need_all_hidden_states: bool = False
    need_all_logits: bool = True


@dataclass
class LocalPyModelOutputs:
    """Field-compatible replica of ``PyModelOutputs``."""

    hidden_states: torch.Tensor
    mtp_target_hidden_states: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# PD CacheStore Publication Contract
# ---------------------------------------------------------------------------


class LocalMemoryCacheStoreWriter:
    """In-memory PD CacheStore publisher implementing exact ``CacheStoreWriter.write`` ABI.

    Copies committed physical block slices of each published ``LayerKVCache`` to
    CPU-pinned/staged transport buffers keyed by ``(layer_id, tag)``.
    """

    def __init__(self, *, record_payload: bool = True) -> None:
        self.record_payload = record_payload
        # (layer_id, tag) -> (tuple(phys_blocks), staged_cpu_tensor)
        self.published: Dict[Tuple[int, str], Tuple[Tuple[int, ...], Optional[torch.Tensor]]] = {}
        self.events: List[Tuple[int, str, Tuple[int, ...]]] = []

    def clear(self) -> None:
        self.published.clear()
        self.events.clear()

    def write(self, cache_store_inputs: Any, kv_cache: Any) -> None:
        tag = str(getattr(cache_store_inputs, "tag", kv_cache.tag))
        req_blocks = tuple(int(b) for b in getattr(cache_store_inputs, "block_ids", ()) if int(b) > 0)
        layer_id = int(kv_cache.layer_id)
        payload: Optional[torch.Tensor] = None
        if self.record_payload:
            if req_blocks:
                idx = torch.tensor(req_blocks, dtype=torch.long, device=kv_cache.kv_cache_base.device)
                payload = kv_cache.kv_cache_base.index_select(0, idx).detach().clone()
            else:
                payload = kv_cache.kv_cache_base.detach().clone()
        self.published[(layer_id, tag)] = (req_blocks, payload)
        self.events.append((layer_id, tag, req_blocks))


def _iter_attention_groups(attention_inputs: Any) -> List[Tuple[Optional[str], Any]]:
    if isinstance(attention_inputs, Mapping):
        return [(str(k), v) for k, v in attention_inputs.items()]
    return [(None, attention_inputs)]


def _extract_input_attr(inputs: Any, name: str, default: Any = None) -> Any:
    if isinstance(inputs, Mapping):
        for v in inputs.values():
            val = getattr(v, name, None)
            if val is not None:
                return val
        return default
    return getattr(inputs, name, default)


def _select_group_for_tag(attention_inputs: Any, tag: str) -> Any:
    if isinstance(attention_inputs, Mapping):
        if tag in attention_inputs:
            return attention_inputs[tag]
        if len(attention_inputs) == 1:
            return next(iter(attention_inputs.values()))
        # Fall back to "full" or first entry if heterogeneous mapping omits auxiliary tag
        if FULL_ATTN_KV_TAG in attention_inputs:
            return attention_inputs[FULL_ATTN_KV_TAG]
        return next(iter(attention_inputs.values()))
    return attention_inputs


def validate_qwen4_5_pd_cache_inputs(inputs: Sequence[Any]) -> bool:
    """Validate whether this forward executes a PD-disaggregated Prefill publication.

    Follows ``qwen4_exp_runtime/pd_publication.py`` strictly:
      - Returns ``False`` when no region carries ``cache_store_inputs`` (ordinary serving / decode).
      - Raises ``RuntimeError`` before any state mutation if any region has a partial plan,
        missing writer, non-prefill flags, CUDA Graph capture, or target-verify mode.
    """
    items = tuple(inputs)
    planned = [getattr(v, "cache_store_inputs", None) is not None for v in items]
    if not any(planned):
        return False
    if not all(planned):
        raise RuntimeError(
            "qwen4_5 PD cache regions have incomplete publication plans"
        )
    for val in items:
        if not bool(getattr(val, "is_prefill", False)):
            raise RuntimeError("qwen4_5 PD publication requires prefill inputs")
        if bool(getattr(val, "is_cuda_graph", False)):
            raise RuntimeError("qwen4_5 PD publication requires eager prefill")
        if bool(getattr(val, "is_target_verify", False)):
            raise RuntimeError(
                "qwen4_5 PD cannot publish speculative target-verify state"
            )
        writer = getattr(val, "cache_store_writer", None)
        if writer is None or not callable(getattr(writer, "write", None)):
            raise RuntimeError("qwen4_5 PD cache-store writer is missing")
    return True


def prepare_qwen4_5_cache_store_writers(
    attention_inputs: Any,
    kv_cache: Any,
    *,
    layer_count: int,
    write_op_factory: Optional[Callable[[Any, Any], Any]] = None,
) -> Dict[str, Any]:
    """Build per-tag publication writers for all active Qwen 4.5-Flash cache regions."""
    groups = _iter_attention_groups(attention_inputs)
    if not validate_qwen4_5_pd_cache_inputs([v for _, v in groups]):
        return {}
    if kv_cache is None:
        raise RuntimeError("qwen4_5 PD publication requires initialized kv_cache")

    required: set[str] = set()
    for lid in range(layer_count):
        for cache in kv_cache.get_layer_cache_groups(lid):
            required.add(str(cache.tag))

    if isinstance(attention_inputs, Mapping):
        missing = required - set(attention_inputs.keys())
        if missing:
            raise RuntimeError(
                f"qwen4_5 PD cache inputs are missing {sorted(missing)}"
            )

    factory = write_op_factory
    if factory is None:
        try:
            from rtp_llm.models_py.modules.factory.attention.common import (
                create_write_cache_store_impl,
            )

            factory = create_write_cache_store_impl
        except Exception:
            factory = None

    writers: Dict[str, Any] = {}
    for tag in sorted(required):
        group_inputs = _select_group_for_tag(attention_inputs, tag)
        cs_inputs = getattr(group_inputs, "cache_store_inputs", None)
        cs_writer = getattr(group_inputs, "cache_store_writer", None)
        if factory is not None and _HAS_COMPILED_GPT_BASE and not isinstance(kv_cache, LocalKVCache):
            op = factory(group_inputs, kv_cache)
            if op is not None:
                writers[tag] = op
                continue
        if cs_writer is None or cs_inputs is None:
            raise RuntimeError(f"qwen4_5 PD cache writer is missing for {tag!r}")
        writers[tag] = (cs_writer, cs_inputs)

    return writers


def publish_qwen4_5_layer_caches(kv_cache: Any, layer_idx: int, writers: Mapping[str, Any]) -> None:
    """Publish all committed caches of ``layer_idx`` immediately after layer state commit."""
    if not writers:
        return
    for cache in kv_cache.get_layer_cache_groups(layer_idx):
        tag = str(cache.tag)
        if tag not in writers:
            raise RuntimeError(f"qwen4_5 PD layer {layer_idx} has no writer for {tag!r}")
        entry = writers[tag]
        if callable(entry):
            entry(cache)
        else:
            writer, cs_inputs = entry
            writer.write(cs_inputs, cache)


# ---------------------------------------------------------------------------
# Paged KVCache State Bridge (Prefill + Prefix Reuse + Cross-Page Decode)
# ---------------------------------------------------------------------------


def _block_table_for_tag(attention_inputs: Any, tag: str) -> torch.Tensor:
    grp = _select_group_for_tag(attention_inputs, tag)
    prefer_kernel = tag == INDEXER_KV_TAG or (
        tag != INDEXER_STATE_TAG and bool(getattr(grp, "is_cuda_graph", False))
    )
    names = (
        (
            "kv_cache_kernel_block_id_device",
            "kv_cache_kernel_block_id",
            "kv_cache_block_id_device",
            "kv_cache_block_id",
        )
        if prefer_kernel
        else (
            "kv_cache_block_id_device",
            "kv_cache_kernel_block_id_device",
            "kv_cache_block_id",
            "kv_cache_kernel_block_id",
        )
    )
    table = None
    for name in names:
        cand = getattr(grp, name, None)
        if cand is not None and cand.numel() > 0:
            table = cand
            break
    if table is None or table.ndim != 2:
        raise RuntimeError(f"qwen4_5 paged cache {tag!r} requires a 2D block table")
    return table


def _is_padded_group(attention_inputs: Any, tag: str) -> bool:
    grp = _select_group_for_tag(attention_inputs, tag)
    return bool(getattr(grp, "is_s_padded", False))


def _check_phys_block(phys_id: int, pool_rows: int, tag: str, layer_idx: int) -> int:
    if phys_id <= 0 or phys_id >= pool_rows:
        raise RuntimeError(
            f"Invalid physical block id {phys_id} for layer {layer_idx} tag={tag!r} (pool_rows={pool_rows})"
        )
    return phys_id


def _reshape_paged_kv_cache(
    paged_kv_cache: torch.Tensor,
    num_kv_heads: int,
    tokens_per_block: int,
    head_dim: int,
) -> torch.Tensor:
    if paged_kv_cache.dim() != 2:
        return paged_kv_cache
    block_num = int(paged_kv_cache.shape[0])
    expected_elems_per_block = 2 * num_kv_heads * tokens_per_block * head_dim
    if int(paged_kv_cache.shape[1]) < expected_elems_per_block:
        raise ValueError(
            f"packed kv_cache_base has insufficient stride: "
            f"got stride={paged_kv_cache.shape[1]} elems, need={expected_elems_per_block} elems"
        )
    return paged_kv_cache[:, :expected_elems_per_block].reshape(
        block_num, 2, num_kv_heads, tokens_per_block, head_dim
    )


def _view_mha_kv_nhd(
    base: torch.Tensor,
    num_kv_heads: int,
    tokens_per_block: int,
    head_dim: int,
) -> torch.Tensor:
    """Return a live zero-copy NHD view ``[num_blocks, 2, B_tok, H_kv, D]`` of MHA pool."""
    if base.ndim == 2:
        need = 2 * num_kv_heads * tokens_per_block * head_dim
        if int(base.shape[1]) < need:
            raise RuntimeError(
                f"MHA packed pool stride too small: {base.shape[1]} < {need}"
            )
        return (
            base[:, :need]
            .reshape(int(base.shape[0]), 2, num_kv_heads, tokens_per_block, head_dim)
            .transpose(2, 3)
        )
    if base.ndim == 5:
        if (
            int(base.shape[2]) == num_kv_heads
            and int(base.shape[3]) == tokens_per_block
            and int(base.shape[4]) == head_dim
        ):
            return base.transpose(2, 3)
        if (
            int(base.shape[2]) == tokens_per_block
            and int(base.shape[3]) == num_kv_heads
            and int(base.shape[4]) == head_dim
        ):
            return base
    raise RuntimeError(
        f"Unexpected MHA kv_cache_base shape {tuple(base.shape)} for "
        f"(H_kv={num_kv_heads}, B_tok={tokens_per_block}, D={head_dim})"
    )


def _view_opaque_row(
    base: torch.Tensor,
    phys_id: int,
    target_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Return a live zero-copy view of physical block ``phys_id`` as ``target_shape``."""
    if (
        base.ndim == 1 + len(target_shape)
        and tuple(int(d) for d in base.shape[1:]) == target_shape
        and base.dtype == target_dtype
    ):
        return base[phys_id]
    elem_size = _dtype_itemsize(target_dtype)
    need_bytes = math.prod(target_shape) * elem_size
    row_u8 = base[phys_id].reshape(-1).view(torch.uint8)
    if int(row_u8.numel()) < need_bytes:
        raise RuntimeError(
            f"Opaque pool row too small: {row_u8.numel()} bytes < {need_bytes} bytes "
            f"for shape={target_shape} dtype={target_dtype}"
        )
    return row_u8[:need_bytes].view(target_dtype).view(*target_shape)


def _cast_for_kv_pool(t: torch.Tensor, pool_dtype: torch.dtype) -> torch.Tensor:
    """Cast K/V tensor to KV pool dtype, saturating cleanly to E4M3 range to avoid NaNs."""
    if t.dtype == pool_dtype:
        return t
    if pool_dtype == torch.float8_e4m3fn:
        return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return t.to(pool_dtype)


def _resolve_kv_scale_4d(
    kv_scale: Optional[torch.Tensor],
    num_blocks: int,
    num_kv_heads: int,
    tokens_per_block: int,
) -> Optional[torch.Tensor]:
    """View 4D HND or 2D packed FP8 KV scales as ``[N, 2, H_kv, B_tok]``."""
    if kv_scale is None or kv_scale.dtype != torch.float32:
        return None
    if (
        kv_scale.ndim == 2
        and int(kv_scale.shape[0]) == num_blocks
        and int(kv_scale.shape[1]) >= 2 * num_kv_heads * tokens_per_block
        and int(kv_scale.stride(1)) == 1
    ):
        return torch.as_strided(
            kv_scale,
            (num_blocks, 2, num_kv_heads, tokens_per_block),
            (int(kv_scale.stride(0)), num_kv_heads * tokens_per_block, tokens_per_block, 1),
        )
    if kv_scale.ndim == 4 and tuple(int(d) for d in kv_scale.shape) == (
        num_blocks,
        2,
        num_kv_heads,
        tokens_per_block,
    ):
        return kv_scale
    return None


def _dequantize_full_attn_kv_pages(
    gathered_nhd: torch.Tensor,
    page_ids: torch.Tensor,
    kv_scale: Optional[torch.Tensor],
    *,
    num_blocks: int,
    tokens_per_block: int,
    num_kv_heads: int,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split gathered NHD pages into K/V, dequantizing FP8 E4M3 pools when needed."""
    k = gathered_nhd[..., 0, :, :, :]
    v = gathered_nhd[..., 1, :, :, :]
    if gathered_nhd.dtype != torch.float8_e4m3fn:
        if out_dtype is not None and k.dtype != out_dtype:
            return k.to(out_dtype), v.to(out_dtype)
        return k, v
    target_dtype = out_dtype or torch.bfloat16
    if kv_scale is None:
        return k.to(target_dtype), v.to(target_dtype)
    if kv_scale.ndim == 1 and kv_scale.numel() >= 2:
        k_out = (k.to(torch.float32) * kv_scale[0].to(torch.float32)).to(target_dtype)
        v_out = (v.to(torch.float32) * kv_scale[1].to(torch.float32)).to(target_dtype)
        return k_out, v_out
    sc_4d = _resolve_kv_scale_4d(kv_scale, num_blocks, num_kv_heads, tokens_per_block)
    if sc_4d is None:
        return k.to(target_dtype), v.to(target_dtype)
    sc_g = sc_4d.index_select(0, page_ids)
    # Match batch prefix of gathered_nhd: [..., n_pages, 2, H_kv, B_tok] -> [..., n_pages, 2, B_tok, H_kv, 1]
    batch_prefix = gathered_nhd.shape[:-4]
    sc_nhd = sc_g.view(*batch_prefix, 2, num_kv_heads, tokens_per_block).transpose(-2, -1).unsqueeze(-1)
    k_out = (k.to(torch.float32) * sc_nhd[..., 0, :, :, :]).to(target_dtype)
    v_out = (v.to(torch.float32) * sc_nhd[..., 1, :, :, :]).to(target_dtype)
    return k_out, v_out


def _load_single_req_full_attn_kv(
    base: torch.Tensor,
    blocks: torch.Tensor,
    total_tokens: int,
    tokens_per_block: int,
    *,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    allow_zero_padded: bool = False,
    kv_scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read [total_tokens, H_kv, D] K/V from MHA paged pool (NHD, HND, or 2D packed)."""
    if num_kv_heads is not None and head_dim is not None:
        nhd = _view_mha_kv_nhd(base, num_kv_heads, tokens_per_block, head_dim)
    else:
        nhd = base
        num_kv_heads, head_dim = int(nhd.shape[3]), int(nhd.shape[4])
    ret_dtype = (
        out_dtype
        if out_dtype is not None
        else (torch.bfloat16 if nhd.dtype == torch.float8_e4m3fn else nhd.dtype)
    )
    if total_tokens <= 0:
        empty = torch.empty((0, num_kv_heads, head_dim), dtype=ret_dtype, device=nhd.device)
        return empty, empty
    n_pages = (total_tokens + tokens_per_block - 1) // tokens_per_block
    page_ids = blocks[:n_pages].to(device=nhd.device, dtype=torch.long)
    if not torch.cuda.is_current_stream_capturing():
        if allow_zero_padded and (page_ids == 0).all().item():
            z = torch.zeros((total_tokens, num_kv_heads, head_dim), dtype=ret_dtype, device=nhd.device)
            return z, z.clone()
        if (page_ids <= 0).any().item() or (page_ids >= nhd.shape[0]).any().item():
            raise RuntimeError(
                f"Invalid FullAttn physical page ids {page_ids.tolist()} for pool={nhd.shape[0]}"
            )
    gathered = nhd.index_select(0, page_ids)  # [n_pages, 2, B_tok, H_kv, D]
    k_blk, v_blk = _dequantize_full_attn_kv_pages(
        gathered,
        page_ids,
        kv_scale,
        num_blocks=int(nhd.shape[0]),
        tokens_per_block=tokens_per_block,
        num_kv_heads=num_kv_heads,
        out_dtype=out_dtype,
    )
    k = k_blk.reshape(n_pages * tokens_per_block, num_kv_heads, head_dim)[:total_tokens]
    v = v_blk.reshape(n_pages * tokens_per_block, num_kv_heads, head_dim)[:total_tokens]
    return k, v


def _load_batched_full_attn_kv(
    base: torch.Tensor,
    bt_blocks: torch.Tensor,
    bsz: int,
    total_tokens: int,
    tokens_per_block: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    kv_scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized [bsz, total_tokens, H_kv, D] gather from paged MHA pool."""
    nhd = _view_mha_kv_nhd(base, num_kv_heads, tokens_per_block, head_dim)
    ret_dtype = (
        out_dtype
        if out_dtype is not None
        else (torch.bfloat16 if nhd.dtype == torch.float8_e4m3fn else nhd.dtype)
    )
    if bsz <= 0 or total_tokens <= 0:
        empty = torch.empty((max(0, bsz), max(0, total_tokens), num_kv_heads, head_dim), dtype=ret_dtype, device=nhd.device)
        return empty, empty
    n_pages = (total_tokens + tokens_per_block - 1) // tokens_per_block
    page_ids = bt_blocks[:bsz, :n_pages].reshape(-1).to(device=nhd.device, dtype=torch.long)
    if not torch.cuda.is_current_stream_capturing() and (
        ((page_ids <= 0) | (page_ids >= nhd.shape[0])).any().item()
    ):
        raise RuntimeError(
            f"Invalid FullAttn physical page ids for pool={nhd.shape[0]}"
        )
    gathered = nhd.index_select(0, page_ids).view(
        bsz, n_pages, 2, tokens_per_block, num_kv_heads, head_dim
    )
    k_blk, v_blk = _dequantize_full_attn_kv_pages(
        gathered,
        page_ids,
        kv_scale,
        num_blocks=int(nhd.shape[0]),
        tokens_per_block=tokens_per_block,
        num_kv_heads=num_kv_heads,
        out_dtype=out_dtype,
    )
    k = k_blk.reshape(bsz, n_pages * tokens_per_block, num_kv_heads, head_dim)[:, :total_tokens]
    v = v_blk.reshape(bsz, n_pages * tokens_per_block, num_kv_heads, head_dim)[:, :total_tokens]
    return k, v


def _store_single_req_full_attn_kv(
    base: torch.Tensor,
    blocks: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    start_pos: int,
    tokens_per_block: int,
    *,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    allow_zero_padded: bool = False,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Write [T_new, H_kv, D] K/V starting at ``start_pos`` into paged pool."""
    if num_kv_heads is not None and head_dim is not None:
        nhd = _view_mha_kv_nhd(base, num_kv_heads, tokens_per_block, head_dim)
    else:
        nhd = base
    num_blocks = int(nhd.shape[0])
    h_kv = int(nhd.shape[3])
    sc_4d = _resolve_kv_scale_4d(kv_scale, num_blocks, h_kv, tokens_per_block)
    t_new = int(k.shape[0])
    pos = start_pos
    written = 0
    while written < t_new:
        page_idx = pos // tokens_per_block
        in_page = pos % tokens_per_block
        raw_phys = int(blocks[page_idx].item())
        if allow_zero_padded and raw_phys == 0:
            take = min(tokens_per_block - in_page, t_new - written)
            pos += take
            written += take
            continue
        phys_id = _check_phys_block(raw_phys, num_blocks, FULL_ATTN_KV_TAG, -1)
        take = min(tokens_per_block - in_page, t_new - written)
        nhd[phys_id, 0, in_page : in_page + take].copy_(_cast_for_kv_pool(k[written : written + take], nhd.dtype))
        nhd[phys_id, 1, in_page : in_page + take].copy_(_cast_for_kv_pool(v[written : written + take], nhd.dtype))
        if sc_4d is not None:
            sc_4d[phys_id, :, :, in_page : in_page + take].fill_(1.0)
        pos += take
        written += take


def _unpack_linear_block(
    row_fp32: torch.Tensor,
    ssm_shape: Tuple[int, int, int],
    conv_shape: Tuple[int, int],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ssm_n = ssm_shape[0] * ssm_shape[1] * ssm_shape[2]
    conv_n = conv_shape[0] * conv_shape[1]
    ssm = row_fp32[:ssm_n].view(ssm_shape)
    conv_raw_bf16 = row_fp32[ssm_n:].view(torch.bfloat16)
    conv = conv_raw_bf16[:conv_n].view(conv_shape).to(dtype)
    return ssm, conv


def _pack_linear_block(
    row_fp32: torch.Tensor,
    ssm: torch.Tensor,
    conv: torch.Tensor,
) -> None:
    ssm_flat = ssm.reshape(-1).to(torch.float32)
    ssm_n = ssm_flat.numel()
    row_fp32[:ssm_n].copy_(ssm_flat)
    conv_flat = conv.reshape(-1).to(torch.bfloat16)
    conv_n = conv_flat.numel()
    row_fp32[ssm_n:].view(torch.bfloat16)[:conv_n].copy_(conv_flat)


if _HAS_TRITON:

    @triton.jit
    def _write_opaque_rows_triton(
        dst_ptr,
        src_ptr,
        phys_ids_ptr,
        dst_row_stride: tl.int64,
        src_row_stride: tl.int64,
        n_elems,
        num_blocks: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        phys = tl.load(phys_ids_ptr + row).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        val = tl.load(src_ptr + row.to(tl.int64) * src_row_stride + offs, mask=mask)
        tl.store(dst_ptr + phys.to(tl.int64) * dst_row_stride + offs, val, mask=mask)

    @triton.jit
    def _write_opaque_rows_paged_triton(
        dst_ptr,
        src_ptr,
        block_table_ptr,
        seq_before_ptr,
        dst_row_stride: tl.int64,
        src_row_stride: tl.int64,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        n_elems,
        BLOCK_SIZE: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        pos = tl.load(seq_before_ptr + b).to(tl.int32)
        if pos < 0:
            return
        page = tl.minimum(pos // tpb, max_blocks - 1)
        phys = tl.load(block_table_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        val = tl.load(src_ptr + b * src_row_stride + offs, mask=mask)
        tl.store(dst_ptr + phys.to(tl.int64) * dst_row_stride + offs, val, mask=mask)

    @triton.jit
    def _store_decode_full_attn_kv_triton(
        pool_ptr,
        scale_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        seq_before_ptr,
        s_blk: tl.int64,
        s_kv: tl.int64,
        s_h: tl.int64,
        s_tok: tl.int64,
        s_d: tl.int64,
        sc_sb: tl.int64,
        sc_skv: tl.int64,
        sc_sh: tl.int64,
        sc_st: tl.int64,
        k_sb: tl.int64,
        k_sh: tl.int64,
        k_sd: tl.int64,
        v_sb: tl.int64,
        v_sh: tl.int64,
        v_sd: tl.int64,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        head_dim: tl.int32,
        BLOCK_D: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        IS_FP8_POOL: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        pos = tl.load(seq_before_ptr + b).to(tl.int32)
        if pos < 0:
            return
        page = pos // tpb
        if page >= max_blocks:
            return
        slot = pos - page * tpb
        phys = tl.load(
            block_table_ptr + b.to(tl.int64) * bt_s0 + page.to(tl.int64) * bt_s1
        ).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        k_val = tl.load(
            k_ptr + b.to(tl.int64) * k_sb + h.to(tl.int64) * k_sh + offs_d * k_sd,
            mask=mask_d,
        )
        v_val = tl.load(
            v_ptr + b.to(tl.int64) * v_sb + h.to(tl.int64) * v_sh + offs_d * v_sd,
            mask=mask_d,
        )
        dst_base = (
            phys.to(tl.int64) * s_blk
            + h.to(tl.int64) * s_h
            + slot.to(tl.int64) * s_tok
            + offs_d * s_d
        )
        if IS_FP8_POOL:
            k_val = tl.minimum(tl.maximum(k_val.to(tl.float32), -448.0), 448.0)
            v_val = tl.minimum(tl.maximum(v_val.to(tl.float32), -448.0), 448.0)
        tl.store(pool_ptr + dst_base, k_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        tl.store(pool_ptr + dst_base + s_kv, v_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        if HAS_SCALE:
            sc_base = phys.to(tl.int64) * sc_sb + h.to(tl.int64) * sc_sh + slot.to(tl.int64) * sc_st
            tl.store(scale_ptr + sc_base, 1.0)
            tl.store(scale_ptr + sc_base + sc_skv, 1.0)


if _HAS_TRITON:

    @triton.jit
    def _store_ragged_full_attn_kv_triton(
        pool_ptr,
        scale_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        tok_req_ptr,
        tok_abs_pos_ptr,
        s_blk: tl.int64,
        s_kv: tl.int64,
        s_tok: tl.int64,
        s_h: tl.int64,
        s_d: tl.int64,
        sc_sb: tl.int64,
        sc_skv: tl.int64,
        sc_sh: tl.int64,
        sc_st: tl.int64,
        k_sn: tl.int64,
        k_sh: tl.int64,
        k_sd: tl.int64,
        v_sn: tl.int64,
        v_sh: tl.int64,
        v_sd: tl.int64,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        head_dim: tl.int32,
        BLOCK_D: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        IS_FP8_POOL: tl.constexpr,
    ):
        tok = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        b = tl.load(tok_req_ptr + tok).to(tl.int64)
        pos = tl.load(tok_abs_pos_ptr + tok).to(tl.int32)
        if pos < 0:
            return
        page = pos // tpb
        if page >= max_blocks:
            return
        slot = pos - page * tpb
        phys = tl.load(block_table_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        k_val = tl.load(k_ptr + tok * k_sn + h * k_sh + offs_d * k_sd, mask=mask_d)
        v_val = tl.load(v_ptr + tok * v_sn + h * v_sh + offs_d * v_sd, mask=mask_d)
        dst_base = (
            phys.to(tl.int64) * s_blk
            + slot.to(tl.int64) * s_tok
            + h * s_h
            + offs_d * s_d
        )
        if IS_FP8_POOL:
            k_val = tl.minimum(tl.maximum(k_val.to(tl.float32), -448.0), 448.0)
            v_val = tl.minimum(tl.maximum(v_val.to(tl.float32), -448.0), 448.0)
        tl.store(pool_ptr + dst_base, k_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        tl.store(pool_ptr + dst_base + s_kv, v_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        if HAS_SCALE:
            sc_base = phys.to(tl.int64) * sc_sb + h * sc_sh + slot.to(tl.int64) * sc_st
            tl.store(scale_ptr + sc_base, 1.0)
            tl.store(scale_ptr + sc_base + sc_skv, 1.0)

    @triton.jit
    def _store_verify_full_attn_kv_triton(
        pool_ptr,
        scale_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        seq_before_ptr,
        s_blk: tl.int64,
        s_kv: tl.int64,
        s_h: tl.int64,
        s_tok: tl.int64,
        s_d: tl.int64,
        sc_sb: tl.int64,
        sc_skv: tl.int64,
        sc_sh: tl.int64,
        sc_st: tl.int64,
        k_sb: tl.int64,
        k_sv: tl.int64,
        k_sh: tl.int64,
        k_sd: tl.int64,
        v_sb: tl.int64,
        v_sv: tl.int64,
        v_sh: tl.int64,
        v_sd: tl.int64,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        head_dim: tl.int32,
        BLOCK_D: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        IS_FP8_POOL: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        v_step = tl.program_id(1).to(tl.int64)
        h = tl.program_id(2).to(tl.int64)
        pos = tl.load(seq_before_ptr + b).to(tl.int32) + v_step.to(tl.int32)
        if pos < 0:
            return
        page = pos // tpb
        if page >= max_blocks:
            return
        slot = pos - page * tpb
        phys = tl.load(block_table_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        k_val = tl.load(
            k_ptr + b * k_sb + v_step * k_sv + h * k_sh + offs_d * k_sd, mask=mask_d
        )
        v_val = tl.load(
            v_ptr + b * v_sb + v_step * v_sv + h * v_sh + offs_d * v_sd, mask=mask_d
        )
        dst_base = (
            phys.to(tl.int64) * s_blk
            + h * s_h
            + slot.to(tl.int64) * s_tok
            + offs_d * s_d
        )
        if IS_FP8_POOL:
            k_val = tl.minimum(tl.maximum(k_val.to(tl.float32), -448.0), 448.0)
            v_val = tl.minimum(tl.maximum(v_val.to(tl.float32), -448.0), 448.0)
        tl.store(pool_ptr + dst_base, k_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        tl.store(pool_ptr + dst_base + s_kv, v_val.to(pool_ptr.dtype.element_ty), mask=mask_d)
        if HAS_SCALE:
            sc_base = phys.to(tl.int64) * sc_sb + h * sc_sh + slot.to(tl.int64) * sc_st
            tl.store(scale_ptr + sc_base, 1.0)
            tl.store(scale_ptr + sc_base + sc_skv, 1.0)

    @triton.jit
    def _read_opaque_rows_triton(
        dst_ptr,
        src_ptr,
        block_table_ptr,
        seq_before_ptr,
        dst_row_stride: tl.int64,
        src_row_stride: tl.int64,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        n_elems,
        fill_val,
        BLOCK_SIZE: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        pos = tl.load(seq_before_ptr + b).to(tl.int32)
        page = tl.minimum(tl.maximum((pos - 1) // tpb, 0), max_blocks - 1)
        phys = tl.load(block_table_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        valid = (pos > 0) & (phys > 0) & (phys < num_blocks)
        safe_phys = tl.minimum(tl.maximum(phys, 0), num_blocks - 1).to(tl.int64)
        val = tl.load(
            src_ptr + safe_phys * src_row_stride + offs,
            mask=mask & valid,
            other=fill_val,
        )
        tl.store(dst_ptr + b * dst_row_stride + offs, val, mask=mask)

    @triton.jit
    def _prepare_opaque_tx_commit_triton(
        bt_ptr,
        pos_before_ptr,
        accept_len_ptr,
        steps_ptr,
        base_ptr,
        phys_ids_ptr,
        commit_ptr,
        undo_ptr,
        bt_s0: tl.int64,
        bt_s1: tl.int64,
        steps_sb: tl.int64,
        steps_sv: tl.int64,
        base_sb: tl.int64,
        commit_sb: tl.int64,
        undo_sb: tl.int64,
        live_bs: tl.int32,
        verify_w: tl.int32,
        tpb: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        n_elems,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1).to(tl.int64)
        is_bnd = row >= live_bs
        b = row - live_bs if is_bnd else row
        a_len = tl.load(accept_len_ptr + b).to(tl.int32)
        pos_b = tl.load(pos_before_ptr + b).to(tl.int32)
        valid = a_len > 0

        if not is_bnd:
            step_idx = tl.minimum(tl.maximum(a_len - 1, 0), verify_w - 1)
            target_pos = tl.maximum(pos_b + a_len - 1, 0)
            do_commit = valid
        else:
            boundary_step = (tpb - 1) - (pos_b % tpb)
            crossed = (
                valid
                & ((a_len - 1) > boundary_step)
                & (boundary_step >= 0)
                & (boundary_step < verify_w)
            )
            step_idx = tl.minimum(tl.maximum(boundary_step, 0), verify_w - 1)
            target_pos = pos_b
            do_commit = crossed

        page = tl.minimum(tl.maximum(target_pos // tpb, 0), max_blocks - 1)
        raw_phys = tl.load(bt_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        phys = tl.where(do_commit & (raw_phys > 0) & (raw_phys < num_blocks), raw_phys, 0)
        if blk == 0:
            tl.store(phys_ids_ptr + row, phys)
        if phys <= 0:
            return

        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        c_val = tl.load(
            steps_ptr + b * steps_sb + step_idx.to(tl.int64) * steps_sv + offs,
            mask=mask,
        )
        tl.store(commit_ptr + row * commit_sb + offs, c_val, mask=mask)
        u_val = tl.load(base_ptr + phys.to(tl.int64) * base_sb + offs, mask=mask)
        tl.store(undo_ptr + row * undo_sb + offs, u_val, mask=mask)

    @triton.jit
    def _prepare_opaque_tx_commit_batched_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_bt_ptr,
        ptrs_steps_ptr,
        ptrs_base_ptr,
        ptrs_phys_ptr,
        ptrs_commit_ptr,
        ptrs_undo_ptr,
        pos_before_ptr,
        accept_len_ptr,
        bt_s0_ptr,
        bt_s1_ptr,
        steps_sb_ptr,
        steps_sv_ptr,
        base_sb_ptr,
        commit_sb_ptr,
        undo_sb_ptr,
        tpb_ptr,
        max_blocks_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        live_bs: tl.int32,
        verify_w: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Prepare commit & undo snapshots across compact active tiles in 1 kernel."""
        tile_idx = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        tpb = tl.load(tpb_ptr + e).to(tl.int32)
        max_blocks = tl.load(max_blocks_ptr + e).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        is_bnd = row >= live_bs
        b = row - live_bs if is_bnd else row
        a_len = tl.load(accept_len_ptr + b).to(tl.int32)
        pos_b = tl.load(pos_before_ptr + b).to(tl.int32)
        valid = a_len > 0

        if not is_bnd:
            step_idx = tl.minimum(tl.maximum(a_len - 1, 0), verify_w - 1)
            target_pos = tl.maximum(pos_b + a_len - 1, 0)
            do_commit = valid
        else:
            boundary_step = (tpb - 1) - (pos_b % tpb)
            crossed = (
                valid
                & ((a_len - 1) > boundary_step)
                & (boundary_step >= 0)
                & (boundary_step < verify_w)
            )
            step_idx = tl.minimum(tl.maximum(boundary_step, 0), verify_w - 1)
            target_pos = pos_b
            do_commit = crossed

        bt_ptr = tl.load(ptrs_bt_ptr + e).to(tl.pointer_type(tl.int32))
        bt_s0 = tl.load(bt_s0_ptr + e).to(tl.int64)
        bt_s1 = tl.load(bt_s1_ptr + e).to(tl.int64)
        page = tl.minimum(tl.maximum(target_pos // tpb, 0), max_blocks - 1)
        raw_phys = tl.load(bt_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        phys = tl.where(do_commit & (raw_phys > 0) & (raw_phys < num_blocks), raw_phys, 0)
        if blk == 0:
            phys_ids_ptr = tl.load(ptrs_phys_ptr + e).to(tl.pointer_type(tl.int32))
            tl.store(phys_ids_ptr + row, phys)
        if phys <= 0:
            return

        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        steps_ptr = tl.load(ptrs_steps_ptr + e).to(tl.pointer_type(tl.int32))
        steps_sb = tl.load(steps_sb_ptr + e).to(tl.int64)
        steps_sv = tl.load(steps_sv_ptr + e).to(tl.int64)
        commit_ptr = tl.load(ptrs_commit_ptr + e).to(tl.pointer_type(tl.int32))
        commit_sb = tl.load(commit_sb_ptr + e).to(tl.int64)
        base_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)
        undo_ptr = tl.load(ptrs_undo_ptr + e).to(tl.pointer_type(tl.int32))
        undo_sb = tl.load(undo_sb_ptr + e).to(tl.int64)

        c_val = tl.load(
            steps_ptr + b * steps_sb + step_idx.to(tl.int64) * steps_sv + offs,
            mask=mask,
        )
        tl.store(commit_ptr + row * commit_sb + offs, c_val, mask=mask)
        u_val = tl.load(base_ptr + phys.to(tl.int64) * base_sb + offs, mask=mask)
        tl.store(undo_ptr + row * undo_sb + offs, u_val, mask=mask)

    @triton.jit
    def _write_opaque_rows_batched_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_dst_ptr,
        ptrs_src_ptr,
        ptrs_phys_ptr,
        dst_sb_ptr,
        src_sb_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter commit or undo snapshots back to persistent opaque pools across compact tiles."""
        tile_idx = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        phys_ptr = tl.load(ptrs_phys_ptr + e).to(tl.pointer_type(tl.int32))
        phys = tl.load(phys_ptr + row).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        src_ptr = tl.load(ptrs_src_ptr + e).to(tl.pointer_type(tl.int32))
        src_sb = tl.load(src_sb_ptr + e).to(tl.int64)
        dst_ptr = tl.load(ptrs_dst_ptr + e).to(tl.pointer_type(tl.int32))
        dst_sb = tl.load(dst_sb_ptr + e).to(tl.int64)
        val = tl.load(src_ptr + row * src_sb + offs, mask=mask)
        tl.store(dst_ptr + phys.to(tl.int64) * dst_sb + offs, val, mask=mask)

    @triton.jit
    def _dec_sp_gather_opaque_batched_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_base_ptr,
        ptrs_bt_ptr,
        base_sb_ptr,
        bt_s0_ptr,
        bt_s1_ptr,
        tpb_ptr,
        max_blocks_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        col_off_ptr,
        seq_ptr,
        send_ptr,
        send_s0: tl.int64,
        b0: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Gather local [b0, b1) opaque decode states into contiguous packed send buffer (CG-safe)."""
        tile_idx = tl.program_id(0).to(tl.int64)
        r_loc = tl.program_id(1).to(tl.int64)
        global_b = b0 + r_loc
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        col_off = tl.load(col_off_ptr + e).to(tl.int64)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems

        pos = tl.load(seq_ptr + global_b).to(tl.int32)
        tpb = tl.load(tpb_ptr + e).to(tl.int32)
        max_blocks = tl.load(max_blocks_ptr + e).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        bt_ptr = tl.load(ptrs_bt_ptr + e).to(tl.pointer_type(tl.int32))
        bt_s0 = tl.load(bt_s0_ptr + e).to(tl.int64)
        bt_s1 = tl.load(bt_s1_ptr + e).to(tl.int64)
        base_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)

        has_bt = max_blocks > 0
        page = tl.minimum(tl.maximum(pos, 0) // tpb, tl.maximum(max_blocks - 1, 0))
        raw_phys = tl.load(bt_ptr + global_b * bt_s0 + page.to(tl.int64) * bt_s1, mask=has_bt, other=0).to(tl.int32)
        valid = (pos >= 0) & has_bt & (raw_phys > 0) & (raw_phys < num_blocks)
        safe_phys = tl.where(valid, raw_phys, 0).to(tl.int64)
        val = tl.load(base_ptr + safe_phys * base_sb + offs, mask=(mask & valid), other=0)
        tl.store(send_ptr + r_loc * send_s0 + col_off + offs, val, mask=mask)

    @triton.jit
    def _dec_sp_scatter_opaque_batched_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_base_ptr,
        ptrs_bt_ptr,
        base_sb_ptr,
        bt_s0_ptr,
        bt_s1_ptr,
        tpb_ptr,
        max_blocks_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        col_off_ptr,
        seq_ptr,
        all_ptr,
        all_s0: tl.int64,
        b0: tl.int32,
        b1: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter remote SP ranks opaque decode states back to persistent KV pools (CG-safe)."""
        tile_idx = tl.program_id(0).to(tl.int64)
        global_b = tl.program_id(1).to(tl.int64)
        if (global_b >= b0) and (global_b < b1):
            return
        pos = tl.load(seq_ptr + global_b).to(tl.int32)
        if pos < 0:
            return
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        max_blocks = tl.load(max_blocks_ptr + e).to(tl.int32)
        if max_blocks <= 0:
            return
        tpb = tl.load(tpb_ptr + e).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        bt_ptr = tl.load(ptrs_bt_ptr + e).to(tl.pointer_type(tl.int32))
        bt_s0 = tl.load(bt_s0_ptr + e).to(tl.int64)
        bt_s1 = tl.load(bt_s1_ptr + e).to(tl.int64)
        page = tl.minimum(pos // tpb, max_blocks - 1)
        phys = tl.load(bt_ptr + global_b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        col_off = tl.load(col_off_ptr + e).to(tl.int64)
        base_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)
        val = tl.load(all_ptr + global_b * all_s0 + col_off + offs, mask=mask)
        tl.store(base_ptr + phys.to(tl.int64) * base_sb + offs, val, mask=mask)

    @triton.jit
    def _prefill_sp_gather_opaque_batched_triton(
        tile_e_ptr,
        tile_slot_ptr,
        tile_blk_ptr,
        tile_off_ptr,
        ptrs_base_ptr,
        ptrs_sphys_ptr,
        base_sb_ptr,
        sphys_s0_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        send_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Batched Triton gather of local SP prefill opaque snapshots into packed send buffer."""
        tile_idx = tl.program_id(0).to(tl.int64)
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        slot = tl.load(tile_slot_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        out_off = tl.load(tile_off_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        sphys_ptr = tl.load(ptrs_sphys_ptr + e).to(tl.pointer_type(tl.int32))
        sphys_s0 = tl.load(sphys_s0_ptr + e).to(tl.int64)
        phys = tl.load(sphys_ptr + slot * sphys_s0).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        valid = (phys > 0) & (phys < num_blocks)
        safe_phys = tl.where(valid, phys, 0).to(tl.int64)
        base_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)
        val = tl.load(base_ptr + safe_phys * base_sb + offs, mask=(mask & valid), other=0)
        tl.store(send_ptr + out_off + offs, val, mask=mask)

    @triton.jit
    def _prefill_sp_scatter_opaque_batched_triton(
        tile_e_ptr,
        tile_slot_ptr,
        tile_blk_ptr,
        tile_rank_ptr,
        tile_off_ptr,
        ptrs_base_ptr,
        ptrs_sphys_ptr,
        base_sb_ptr,
        sphys_s0_ptr,
        num_blocks_ptr,
        n_elems_ptr,
        all_ptr,
        all_s0: tl.int64,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Batched Triton scatter of remote SP prefill opaque snapshots into local persistent pools."""
        tile_idx = tl.program_id(0).to(tl.int64)
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        slot = tl.load(tile_slot_ptr + tile_idx).to(tl.int64)
        sphys_ptr = tl.load(ptrs_sphys_ptr + e).to(tl.pointer_type(tl.int32))
        sphys_s0 = tl.load(sphys_s0_ptr + e).to(tl.int64)
        phys = tl.load(sphys_ptr + slot * sphys_s0).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        if phys <= 0 or phys >= num_blocks:
            return
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        r_i = tl.load(tile_rank_ptr + tile_idx).to(tl.int64)
        in_off = tl.load(tile_off_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        base_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)
        val = tl.load(all_ptr + r_i * all_s0 + in_off + offs, mask=mask)
        tl.store(base_ptr + phys.to(tl.int64) * base_sb + offs, val, mask=mask)

    @triton.jit
    def _tx_sp_batch_pack_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_commit_ptr,
        commit_sb_ptr,
        n_elems_ptr,
        unit_off_ptr,
        send_ptr,
        lo_m: tl.int64,
        cnt_m: tl.int64,
        live_bs: tl.int64,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Pack local [lo_m, hi_m) commit & undo rows across all SP records into contiguous AllGather buffer."""
        tile_idx = tl.program_id(0).to(tl.int64)
        idx = tl.program_id(1).to(tl.int64)
        half = tl.program_id(2).to(tl.int64)
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        c_ptr = tl.load(ptrs_commit_ptr + e).to(tl.pointer_type(tl.int32))
        c_sb = tl.load(commit_sb_ptr + e).to(tl.int64)
        u_off = tl.load(unit_off_ptr + e).to(tl.int64)
        row_c = half * live_bs + lo_m + idx
        out_off = u_off * cnt_m + (half * cnt_m + idx) * n_elems.to(tl.int64) + offs
        val = tl.load(c_ptr + row_c * c_sb + offs, mask=mask)
        tl.store(send_ptr + out_off, val, mask=mask)

    @triton.jit
    def _tx_sp_batch_unpack_triton(
        tile_e_ptr,
        tile_blk_ptr,
        ptrs_commit_ptr,
        commit_sb_ptr,
        n_elems_ptr,
        unit_off_ptr,
        r_lo_ptr,
        r_hi_ptr,
        all_ptr,
        all_s0: tl.int64,
        live_bs: tl.int64,
        eff_rk: tl.int64,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Unpack remote SP rank commit & undo rows back into stage commit_u8 buffers."""
        tile_idx = tl.program_id(0).to(tl.int64)
        idx = tl.program_id(1).to(tl.int64)
        p2 = tl.program_id(2).to(tl.int64)
        r_i = p2 // 2
        if r_i == eff_rk:
            return
        lo_r = tl.load(r_lo_ptr + r_i).to(tl.int64)
        hi_r = tl.load(r_hi_ptr + r_i).to(tl.int64)
        cnt_r = hi_r - lo_r
        if idx >= cnt_r:
            return
        half = p2 % 2
        e = tl.load(tile_e_ptr + tile_idx).to(tl.int64)
        blk = tl.load(tile_blk_ptr + tile_idx).to(tl.int64)
        n_elems = tl.load(n_elems_ptr + e).to(tl.int32)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        c_ptr = tl.load(ptrs_commit_ptr + e).to(tl.pointer_type(tl.int32))
        c_sb = tl.load(commit_sb_ptr + e).to(tl.int64)
        u_off = tl.load(unit_off_ptr + e).to(tl.int64)
        row_c = half * live_bs + lo_r + idx
        in_off = u_off * cnt_r + (half * cnt_r + idx) * n_elems.to(tl.int64) + offs
        val = tl.load(all_ptr + r_i * all_s0 + in_off, mask=mask)
        tl.store(c_ptr + row_c * c_sb + offs, val, mask=mask)

    @triton.jit
    def _prepare_gdn_ssm_lazy_commit_batched_triton(
        sample_kv_ptr,
        ptrs_bt_ptr,
        ptrs_init_ptr,
        ptrs_k_ptr,
        ptrs_v_ptr,
        ptrs_g_ptr,
        ptrs_beta_ptr,
        ptrs_base_ptr,
        ptrs_phys_ptr,
        ptrs_commit_ptr,
        ptrs_undo_ptr,
        pos_before_ptr,
        accept_len_ptr,
        bt_s0_ptr,
        bt_s1_ptr,
        init_sb_ptr,
        k_sb_ptr,
        k_sv_ptr,
        k_shk_ptr,
        k_sd_ptr,
        v_sb_ptr,
        v_sv_ptr,
        v_shv_ptr,
        v_vd_ptr,
        g_sb_ptr,
        g_sv_ptr,
        g_shv_ptr,
        b_sb_ptr,
        b_sv_ptr,
        b_shv_ptr,
        base_sb_ptr,
        commit_sb_ptr,
        undo_sb_ptr,
        tpb_ptr,
        max_blocks_ptr,
        num_blocks_ptr,
        live_bs: tl.int32,
        V_STEPS: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        H_V: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        L2_EPS: tl.constexpr,
    ):
        """Lazy GDN SSM replay on accepted/crossed rows only, eliminating forward stage_ssm writes."""
        pid0 = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1).to(tl.int64)
        n_dv: tl.constexpr = D_V // BLOCK_DV
        tiles_per_e: tl.constexpr = H_V * n_dv
        e = pid0 // tiles_per_e
        rem = pid0 % tiles_per_e
        h_v = rem // n_dv
        dv_blk = rem % n_dv
        h_k = h_v // GQA_RATIO

        tpb = tl.load(tpb_ptr + e).to(tl.int32)
        max_blocks = tl.load(max_blocks_ptr + e).to(tl.int32)
        num_blocks = tl.load(num_blocks_ptr + e).to(tl.int32)
        is_bnd = row >= live_bs
        b = row - live_bs if is_bnd else row
        a_len = tl.load(accept_len_ptr + b).to(tl.int32)
        pos_b = tl.load(pos_before_ptr + b).to(tl.int32)
        valid = a_len > 0

        if not is_bnd:
            step_idx = tl.minimum(tl.maximum(a_len - 1, 0), V_STEPS - 1)
            target_pos = tl.maximum(pos_b + a_len - 1, 0)
            do_commit = valid
        else:
            boundary_step = (tpb - 1) - (pos_b % tpb)
            crossed = (
                valid
                & ((a_len - 1) > boundary_step)
                & (boundary_step >= 0)
                & (boundary_step < V_STEPS)
            )
            step_idx = tl.minimum(tl.maximum(boundary_step, 0), V_STEPS - 1)
            target_pos = pos_b
            do_commit = crossed

        bt_ptr = tl.load(ptrs_bt_ptr + e).to(tl.pointer_type(tl.int32))
        bt_s0 = tl.load(bt_s0_ptr + e).to(tl.int64)
        bt_s1 = tl.load(bt_s1_ptr + e).to(tl.int64)
        page = tl.minimum(tl.maximum(target_pos // tpb, 0), max_blocks - 1)
        raw_phys = tl.load(bt_ptr + b * bt_s0 + page.to(tl.int64) * bt_s1).to(tl.int32)
        phys = tl.where(do_commit & (raw_phys > 0) & (raw_phys < num_blocks), raw_phys, 0)
        if rem == 0:
            phys_ids_ptr = tl.load(ptrs_phys_ptr + e).to(tl.pointer_type(tl.int32))
            tl.store(phys_ids_ptr + row, phys)
        if phys <= 0:
            return

        offs_k = tl.arange(0, D_K)
        offs_v = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)
        ssm_offs = h_v * (D_K * D_V) + offs_k[:, None] * D_V + offs_v[None, :]

        base_i32_ptr = tl.load(ptrs_base_ptr + e).to(tl.pointer_type(tl.int32))
        base_sb = tl.load(base_sb_ptr + e).to(tl.int64)
        undo_i32_ptr = tl.load(ptrs_undo_ptr + e).to(tl.pointer_type(tl.int32))
        undo_sb = tl.load(undo_sb_ptr + e).to(tl.int64)
        u_val = tl.load(base_i32_ptr + phys.to(tl.int64) * base_sb + ssm_offs)
        tl.store(undo_i32_ptr + row * undo_sb + ssm_offs, u_val)

        init_ptr = tl.load(ptrs_init_ptr + e).to(tl.pointer_type(tl.float32))
        init_sb = tl.load(init_sb_ptr + e).to(tl.int64)
        h = tl.load(init_ptr + b * init_sb + ssm_offs).to(tl.float32)

        kv_ty: tl.constexpr = sample_kv_ptr.dtype.element_ty
        k_ptr = tl.load(ptrs_k_ptr + e).to(tl.pointer_type(kv_ty))
        v_ptr = tl.load(ptrs_v_ptr + e).to(tl.pointer_type(kv_ty))
        g_ptr = tl.load(ptrs_g_ptr + e).to(tl.pointer_type(tl.float32))
        beta_ptr = tl.load(ptrs_beta_ptr + e).to(tl.pointer_type(tl.float32))
        k_base = (
            k_ptr
            + b * tl.load(k_sb_ptr + e).to(tl.int64)
            + h_k * tl.load(k_shk_ptr + e).to(tl.int64)
            + offs_k * tl.load(k_sd_ptr + e).to(tl.int64)
        )
        v_base = (
            v_ptr
            + b * tl.load(v_sb_ptr + e).to(tl.int64)
            + h_v * tl.load(v_shv_ptr + e).to(tl.int64)
            + offs_v * tl.load(v_vd_ptr + e).to(tl.int64)
        )
        g_base = (
            g_ptr
            + b * tl.load(g_sb_ptr + e).to(tl.int64)
            + h_v * tl.load(g_shv_ptr + e).to(tl.int64)
        )
        beta_base = (
            beta_ptr
            + b * tl.load(b_sb_ptr + e).to(tl.int64)
            + h_v * tl.load(b_shv_ptr + e).to(tl.int64)
        )
        k_sv = tl.load(k_sv_ptr + e).to(tl.int64)
        v_sv = tl.load(v_sv_ptr + e).to(tl.int64)
        g_sv = tl.load(g_sv_ptr + e).to(tl.int64)
        b_sv = tl.load(b_sv_ptr + e).to(tl.int64)

        t = 0
        while t <= step_idx:
            k_t = tl.load(k_base + t * k_sv).to(tl.float32)
            v_t = tl.load(v_base + t * v_sv).to(tl.float32)
            g_t = tl.load(g_base + t * g_sv).to(tl.float32)
            beta_t = tl.load(beta_base + t * b_sv).to(tl.float32)
            k_t = k_t * tl.rsqrt(tl.sum(k_t * k_t, axis=0) + L2_EPS)
            h = h * tl.exp(g_t)
            v_pred = tl.sum(h * k_t[:, None], axis=0)
            delta = (v_t - v_pred) * beta_t
            h = h + k_t[:, None] * delta[None, :]
            t += 1

        commit_f32_ptr = tl.load(ptrs_commit_ptr + e).to(tl.pointer_type(tl.float32))
        commit_sb = tl.load(commit_sb_ptr + e).to(tl.int64)
        tl.store(commit_f32_ptr + row * commit_sb + ssm_offs, h)


def _view_opaque_pool_2d(base: torch.Tensor, target_dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
    """Zero-copy [num_blocks, row_elems] view of an opaque pool in ``target_dtype``."""
    nb = int(base.shape[0])
    elem_bytes = _dtype_itemsize(target_dtype)
    if base.dtype == target_dtype:
        if base.ndim == 2:
            return base, int(base.shape[1])
        flat = base.reshape(nb, -1)
        return flat, int(flat.shape[1])
    base_u8 = base.reshape(nb, -1).view(torch.uint8)
    row_bytes = int(base_u8.shape[1])
    row_stride_bytes = int(base_u8.stride(0))
    if row_bytes % elem_bytes != 0 or row_stride_bytes % elem_bytes != 0 or base_u8.data_ptr() % elem_bytes != 0:
        raise RuntimeError(
            f"Opaque pool alignment mismatch for {target_dtype}: row_bytes={row_bytes}, stride={row_stride_bytes}"
        )
    row_elems = row_bytes // elem_bytes
    row_stride_elems = row_stride_bytes // elem_bytes
    if base_u8.is_contiguous():
        return base_u8.view(target_dtype), row_elems
    viewed = torch.as_strided(
        base_u8.view(target_dtype),
        size=(nb, row_elems),
        stride=(row_stride_elems, 1),
    )
    return viewed, row_elems


def _store_ragged_full_attn_kv_gpu(
    kv_cache_layer: Any,
    block_table: torch.Tensor,
    tok_req_dev: torch.Tensor,
    tok_abs_pos_dev: torch.Tensor,
    k_3d: torch.Tensor,
    v_3d: torch.Tensor,
) -> None:
    tpb = int(kv_cache_layer.seq_size_per_block)
    n_tok, h_kv, head_dim = int(k_3d.shape[0]), int(k_3d.shape[1]), int(k_3d.shape[2])
    max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if n_tok == 0 or max_blocks == 0:
        return
    if not (_HAS_TRITON and k_3d.is_cuda):
        req_list = tok_req_dev.tolist()
        pos_list = tok_abs_pos_dev.tolist()
        nhd = _view_mha_kv_nhd(kv_cache_layer.kv_cache_base, h_kv, tpb, head_dim)
        sc_4d = _resolve_kv_scale_4d(
            getattr(kv_cache_layer, "kv_scale_base", None), int(nhd.shape[0]), h_kv, tpb
        )
        for i in range(n_tok):
            b = int(req_list[i])
            pos = int(pos_list[i])
            if pos < 0:
                continue
            pg = pos // tpb
            sl = pos % tpb
            phys = _check_phys_block(int(block_table[b, pg].item()), int(nhd.shape[0]), FULL_ATTN_KV_TAG, -1)
            nhd[phys, 0, sl].copy_(_cast_for_kv_pool(k_3d[i], nhd.dtype))
            nhd[phys, 1, sl].copy_(_cast_for_kv_pool(v_3d[i], nhd.dtype))
            if sc_4d is not None:
                sc_4d[phys, :, :, sl].fill_(1.0)
        return
    pool_nhd = _view_mha_kv_nhd(kv_cache_layer.kv_cache_base, h_kv, tpb, head_dim)
    num_blocks = int(pool_nhd.shape[0])
    if num_blocks == 0:
        return
    sc_4d = _resolve_kv_scale_4d(
        getattr(kv_cache_layer, "kv_scale_base", None), num_blocks, h_kv, tpb
    )
    has_scale = sc_4d is not None
    sc_ptr = sc_4d if sc_4d is not None else pool_nhd
    sc_sb, sc_skv, sc_sh, sc_st = (
        (int(sc_4d.stride(0)), int(sc_4d.stride(1)), int(sc_4d.stride(2)), int(sc_4d.stride(3)))
        if sc_4d is not None
        else (0, 0, 0, 0)
    )
    s_blk, s_kv, s_tok, s_h, s_d = [int(s) for s in pool_nhd.stride()]
    block_d = max(16, int(triton.next_power_of_2(head_dim)))
    ks0, ks1, ks2 = int(k_3d.stride(0)), int(k_3d.stride(1)), int(k_3d.stride(2))
    vs0, vs1, vs2 = int(v_3d.stride(0)), int(v_3d.stride(1)), int(v_3d.stride(2))
    bt0, bt1 = int(block_table.stride(0)), int(block_table.stride(1))
    is_fp8 = pool_nhd.dtype == torch.float8_e4m3fn
    rt_args = (
        pool_nhd, sc_ptr, k_3d, v_3d, block_table, tok_req_dev, tok_abs_pos_dev,
        s_blk, s_kv, s_tok, s_h, s_d, sc_sb, sc_skv, sc_sh, sc_st,
        ks0, ks1, ks2, vs0, vs1, vs2, bt0, bt1, tpb, max_blocks, num_blocks, head_dim,
    )
    ptr_or = (
        pool_nhd.data_ptr() | sc_ptr.data_ptr() | k_3d.data_ptr() | v_3d.data_ptr()
        | block_table.data_ptr() | tok_req_dev.data_ptr() | tok_abs_pos_dev.data_ptr()
    )
    ckey = (
        pool_nhd.device.index, "rt_store_rag", pool_nhd.dtype, sc_ptr.dtype, k_3d.dtype, v_3d.dtype,
        block_table.dtype, tok_req_dev.dtype, tok_abs_pos_dev.dtype,
        tuple(_triton_int_spec(x) for x in rt_args[7:]),
        block_d, has_scale, is_fp8,
    )
    triton_fast_launch(
        _RUNTIME_TRITON_CACHE, ckey, _store_ragged_full_attn_kv_triton,
        n_tok, h_kv, 1, ptr_or, rt_args,
        {"BLOCK_D": block_d, "HAS_SCALE": has_scale, "IS_FP8_POOL": is_fp8},
        num_warps=4,
    )


def _store_verify_full_attn_kv_gpu(
    kv_cache_layer: Any,
    block_table: torch.Tensor,
    seq_before_dev: torch.Tensor,
    k_4d: torch.Tensor,
    v_4d: torch.Tensor,
) -> None:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU verify paged KV store")

    tpb = int(kv_cache_layer.seq_size_per_block)
    bsz, verify_w, h_kv, head_dim = [int(x) for x in k_4d.shape]
    max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if bsz == 0 or verify_w == 0 or max_blocks == 0:
        return
    pool_hnd = _reshape_paged_kv_cache(kv_cache_layer.kv_cache_base, h_kv, tpb, head_dim)
    if int(pool_hnd.shape[2]) != h_kv or int(pool_hnd.shape[3]) != tpb:
        pool_hnd = pool_hnd.transpose(2, 3)
    num_blocks = int(pool_hnd.shape[0])
    if num_blocks == 0:
        return
    sc_4d = _resolve_kv_scale_4d(
        getattr(kv_cache_layer, "kv_scale_base", None), num_blocks, h_kv, tpb
    )
    has_scale = sc_4d is not None
    sc_ptr = sc_4d if sc_4d is not None else pool_hnd
    sc_sb, sc_skv, sc_sh, sc_st = (
        (int(sc_4d.stride(0)), int(sc_4d.stride(1)), int(sc_4d.stride(2)), int(sc_4d.stride(3)))
        if sc_4d is not None
        else (0, 0, 0, 0)
    )
    s_blk, s_kv, s_h, s_tok, s_d = [int(s) for s in pool_hnd.stride()]
    block_d = max(16, int(triton.next_power_of_2(head_dim)))
    ks0, ks1, ks2, ks3 = int(k_4d.stride(0)), int(k_4d.stride(1)), int(k_4d.stride(2)), int(k_4d.stride(3))
    vs0, vs1, vs2, vs3 = int(v_4d.stride(0)), int(v_4d.stride(1)), int(v_4d.stride(2)), int(v_4d.stride(3))
    bt0, bt1 = int(block_table.stride(0)), int(block_table.stride(1))
    is_fp8 = pool_hnd.dtype == torch.float8_e4m3fn
    rt_args = (
        pool_hnd, sc_ptr, k_4d, v_4d, block_table, seq_before_dev,
        s_blk, s_kv, s_h, s_tok, s_d, sc_sb, sc_skv, sc_sh, sc_st,
        ks0, ks1, ks2, ks3, vs0, vs1, vs2, vs3, bt0, bt1, tpb, max_blocks, num_blocks, head_dim,
    )
    ptr_or = (
        pool_hnd.data_ptr() | sc_ptr.data_ptr() | k_4d.data_ptr() | v_4d.data_ptr()
        | block_table.data_ptr() | seq_before_dev.data_ptr()
    )
    ckey = (
        pool_hnd.device.index, "rt_store_vrf", pool_hnd.dtype, sc_ptr.dtype, k_4d.dtype, v_4d.dtype,
        block_table.dtype, seq_before_dev.dtype,
        tuple(_triton_int_spec(x) for x in rt_args[6:]),
        block_d, has_scale, is_fp8,
    )
    triton_fast_launch(
        _RUNTIME_TRITON_CACHE, ckey, _store_verify_full_attn_kv_triton,
        bsz, verify_w, h_kv, ptr_or, rt_args,
        {"BLOCK_D": block_d, "HAS_SCALE": has_scale, "IS_FP8_POOL": is_fp8},
        num_warps=4,
    )


def _store_decode_full_attn_kv_gpu(
    kv_cache_layer: Any,
    block_table: torch.Tensor,
    seq_before_dev: torch.Tensor,
    k_4d: torch.Tensor,
    v_4d: torch.Tensor,
) -> None:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU paged KV store")

    tpb = int(kv_cache_layer.seq_size_per_block)
    bsz, _, h_kv, head_dim = k_4d.shape
    max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if bsz == 0 or max_blocks == 0:
        return
    pool_hnd = _reshape_paged_kv_cache(kv_cache_layer.kv_cache_base, h_kv, tpb, head_dim)
    if int(pool_hnd.shape[2]) != h_kv or int(pool_hnd.shape[3]) != tpb:
        pool_hnd = pool_hnd.transpose(2, 3)
    num_blocks = int(pool_hnd.shape[0])
    sc_4d = _resolve_kv_scale_4d(
        getattr(kv_cache_layer, "kv_scale_base", None), num_blocks, h_kv, tpb
    )
    has_scale = sc_4d is not None
    sc_ptr = sc_4d if sc_4d is not None else pool_hnd
    sc_sb, sc_skv, sc_sh, sc_st = (
        (int(sc_4d.stride(0)), int(sc_4d.stride(1)), int(sc_4d.stride(2)), int(sc_4d.stride(3)))
        if sc_4d is not None
        else (0, 0, 0, 0)
    )
    k_3d = k_4d[:, 0]
    v_3d = v_4d[:, 0]
    s_blk, s_kv, s_h, s_tok, s_d = pool_hnd.stride()
    block_d = int(triton.next_power_of_2(head_dim))
    ks0, ks1, ks2 = int(k_3d.stride(0)), int(k_3d.stride(1)), int(k_3d.stride(2))
    vs0, vs1, vs2 = int(v_3d.stride(0)), int(v_3d.stride(1)), int(v_3d.stride(2))
    bt0, bt1 = int(block_table.stride(0)), int(block_table.stride(1))
    is_fp8 = pool_hnd.dtype == torch.float8_e4m3fn
    rt_args = (
        pool_hnd, sc_ptr, k_3d, v_3d, block_table, seq_before_dev,
        s_blk, s_kv, s_h, s_tok, s_d, sc_sb, sc_skv, sc_sh, sc_st,
        ks0, ks1, ks2, vs0, vs1, vs2, bt0, bt1, tpb, max_blocks, num_blocks, head_dim,
    )
    ptr_or = (
        pool_hnd.data_ptr() | sc_ptr.data_ptr() | k_3d.data_ptr() | v_3d.data_ptr()
        | block_table.data_ptr() | seq_before_dev.data_ptr()
    )
    ckey = (
        pool_hnd.device.index, "rt_store_dec", pool_hnd.dtype, sc_ptr.dtype, k_3d.dtype, v_3d.dtype,
        block_table.dtype, seq_before_dev.dtype,
        tuple(_triton_int_spec(x) for x in rt_args[6:]),
        block_d, has_scale, is_fp8,
    )
    triton_fast_launch(
        _RUNTIME_TRITON_CACHE, ckey, _store_decode_full_attn_kv_triton,
        bsz, h_kv, 1, ptr_or, rt_args,
        {"BLOCK_D": block_d, "HAS_SCALE": has_scale, "IS_FP8_POOL": is_fp8},
        num_warps=4,
    )


if _HAS_TRITON:

    @triton.jit
    def _qwen4_5_paged_causal_step_attn_triton(
        q_ptr,
        kv_ptr,
        bt_ptr,
        kv_lens_ptr,
        active_ptr,
        out_ptr,
        sm_scale,
        bsz: tl.int32,
        num_kv_heads: tl.int32,
        gqa_groups: tl.int32,
        page_size: tl.int32,
        head_dim: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        q_sb: tl.int64,
        q_sh: tl.int64,
        q_sd: tl.int64,
        o_sb: tl.int64,
        o_sh: tl.int64,
        o_sd: tl.int64,
        bt_sb: tl.int64,
        bt_sc: tl.int64,
        kv_sp: tl.int64,
        kv_skv: tl.int64,
        kv_sh: tl.int64,
        kv_st: tl.int64,
        kv_sd: tl.int64,
        BLOCK_G: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_kv = tl.program_id(1)
        if b >= bsz or h_kv >= num_kv_heads:
            return
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, BLOCK_D)
        mask_g = offs_g < gqa_groups
        mask_d = offs_d < head_dim
        mask_gd = mask_g[:, None] & mask_d[None, :]
        h_q = h_kv * gqa_groups + offs_g
        o_ptrs = (
            out_ptr
            + b.to(tl.int64) * o_sb
            + h_q[:, None].to(tl.int64) * o_sh
            + offs_d[None, :] * o_sd
        )
        if tl.load(active_ptr + b).to(tl.int32) == 0:
            tl.store(
                o_ptrs,
                tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32).to(out_ptr.dtype.element_ty),
                mask=mask_gd,
            )
            return
        kv_len = tl.load(kv_lens_ptr + b).to(tl.int32)
        max_slots = max_blocks * page_size
        if kv_len > max_slots:
            kv_len = max_slots
        if kv_len <= 0:
            tl.store(
                o_ptrs,
                tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32).to(out_ptr.dtype.element_ty),
                mask=mask_gd,
            )
            return

        q_ptrs = (
            q_ptr
            + b.to(tl.int64) * q_sb
            + h_q[:, None].to(tl.int64) * q_sh
            + offs_d[None, :] * q_sd
        )
        q_tile = tl.load(q_ptrs, mask=mask_gd, other=0.0).to(tl.float32)
        m_i = tl.full((BLOCK_G,), -1.0e20, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        start_n = 0
        while start_n < kv_len:
            kv_pos = start_n + offs_n
            valid_n = kv_pos < kv_len
            safe_pos = tl.where(valid_n, kv_pos, 0)
            pages = safe_pos // page_size
            slots = safe_pos - pages * page_size
            phys = tl.load(
                bt_ptr + b.to(tl.int64) * bt_sb + pages.to(tl.int64) * bt_sc,
                mask=valid_n,
                other=0,
            ).to(tl.int32)
            valid_k = valid_n & (phys > 0) & (phys < num_blocks)
            safe_phys = tl.where(valid_k, phys, 0)
            k_base = (
                safe_phys[:, None].to(tl.int64) * kv_sp
                + h_kv.to(tl.int64) * kv_sh
                + slots[:, None].to(tl.int64) * kv_st
                + offs_d[None, :] * kv_sd
            )
            mask_kd = valid_k[:, None] & mask_d[None, :]
            k_tile = tl.load(kv_ptr + k_base, mask=mask_kd, other=0.0).to(tl.float32)
            v_tile = tl.load(
                kv_ptr + k_base + kv_skv, mask=mask_kd, other=0.0
            ).to(tl.float32)
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            qk = tl.where(valid_k[None, :], qk, -1.0e20)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.where(valid_k[None, :], tl.exp(qk - m_ij[:, None]), 0.0)
            alpha = tl.where(m_i > -1.0e19, tl.exp(m_i - m_ij), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v_tile)
            m_i = m_ij
            start_n += BLOCK_N

        out_val = acc / tl.maximum(l_i[:, None], 1.0e-12)
        tl.store(o_ptrs, out_val.to(out_ptr.dtype.element_ty), mask=mask_gd)


def _qwen4_5_paged_causal_step_attn_gpu(
    q_3d: torch.Tensor,
    kv_cache_layer: Any,
    bt_kv: torch.Tensor,
    kv_lens_dev: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU causal paged verify attention")

    bsz, head_num, _ = q_3d.shape
    tpb = int(kv_cache_layer.seq_size_per_block)
    max_blocks = int(bt_kv.shape[1]) if bt_kv.ndim >= 2 else 0
    if bsz == 0 or max_blocks == 0:
        return torch.zeros_like(q_3d)
    pool_hnd = _reshape_paged_kv_cache(kv_cache_layer.kv_cache_base, kv_heads, tpb, head_dim)
    if int(pool_hnd.shape[2]) != kv_heads or int(pool_hnd.shape[3]) != tpb:
        pool_hnd = pool_hnd.transpose(2, 3)
    num_blocks = int(pool_hnd.shape[0])
    if num_blocks == 0:
        return torch.zeros_like(q_3d)
    gqa_groups = head_num // kv_heads
    out = torch.empty_like(q_3d)
    sm_scale = 1.0 / math.sqrt(head_dim)
    block_d = max(16, int(triton.next_power_of_2(head_dim)))
    block_g = max(16, int(triton.next_power_of_2(gqa_groups)))
    _qwen4_5_paged_causal_step_attn_triton[(bsz, kv_heads)](
        q_3d,
        pool_hnd,
        bt_kv,
        kv_lens_dev,
        active_rows,
        out,
        sm_scale,
        bsz,
        kv_heads,
        gqa_groups,
        tpb,
        head_dim,
        max_blocks,
        num_blocks,
        int(q_3d.stride(0)),
        int(q_3d.stride(1)),
        int(q_3d.stride(2)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(bt_kv.stride(0)),
        int(bt_kv.stride(1)),
        int(pool_hnd.stride(0)),
        int(pool_hnd.stride(1)),
        int(pool_hnd.stride(2)),
        int(pool_hnd.stride(3)),
        int(pool_hnd.stride(4)),
        BLOCK_G=block_g,
        BLOCK_N=32,
        BLOCK_D=block_d,
    )
    return out


def _write_opaque_rows_raw_gpu(
    base: torch.Tensor,
    phys_ids_dev: torch.Tensor,
    src_u8: torch.Tensor,
    tag: str,
    layer_idx: int,
) -> None:
    """Write ``[N, need_bytes]`` uint8 rows into persistent opaque ``base`` pool at ``phys_ids_dev > 0``."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU opaque state writes")
    nrows = int(phys_ids_dev.shape[0])
    num_blocks = int(base.shape[0])
    if nrows == 0 or num_blocks == 0:
        return
    need_bytes = int(src_u8.shape[1])
    base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
    if int(base_u8.shape[1]) < need_bytes:
        raise RuntimeError(
            f"Opaque pool row too small for write: {base_u8.shape[1]} bytes < {need_bytes} bytes "
            f"for tag={tag} layer={layer_idx}"
        )
    if (
        need_bytes % 4 == 0
        and int(base_u8.shape[1]) % 4 == 0
        and int(base_u8.stride(0)) % 4 == 0
        and int(src_u8.stride(0)) % 4 == 0
        and base_u8.data_ptr() % 4 == 0
        and src_u8.data_ptr() % 4 == 0
    ):
        dst_view = base_u8.view(torch.int32)
        src_view = src_u8.view(torch.int32)
        n_elems = need_bytes // 4
    else:
        dst_view = base_u8
        src_view = src_u8
        n_elems = need_bytes

    phys_i32 = (
        phys_ids_dev
        if phys_ids_dev.dtype == torch.int32 and phys_ids_dev.is_contiguous()
        else phys_ids_dev.to(torch.int32).contiguous()
    )
    block_sz = 1024
    grid = (nrows, int(triton.cdiv(n_elems, block_sz)))
    dst_s0, src_s0 = int(dst_view.stride(0)), int(src_view.stride(0))
    ptr_or = dst_view.data_ptr() | src_view.data_ptr() | phys_i32.data_ptr()
    ckey = (
        dst_view.device.index, "rt_write_op", dst_view.dtype, src_view.dtype, phys_i32.dtype,
        _triton_int_spec(dst_s0), _triton_int_spec(src_s0), _triton_int_spec(n_elems), _triton_int_spec(num_blocks),
        block_sz,
    )
    triton_fast_launch(
        _RUNTIME_TRITON_CACHE, ckey, _write_opaque_rows_triton,
        grid[0], grid[1], 1, ptr_or,
        (dst_view, src_view, phys_i32, dst_s0, src_s0, n_elems, num_blocks),
        {"BLOCK_SIZE": block_sz},
        num_warps=4,
    )


class Qwen4_5TargetVerifyHandle:
    """CUDA-graph-safe Target-Verify attention handle for dense FullAttn & QSA."""

    def __init__(
        self,
        arch_cfg: Qwen4_5ArchConfig,
        full_attn_inputs: Any,
        indexer_kv_inputs: Any = None,
        indexer_state_inputs: Any = None,
        *,
        page_size: int,
        is_cuda_graph: bool = False,
    ) -> None:
        self.arch_cfg = arch_cfg
        self.full_inputs = full_attn_inputs
        self.indexer_kv_inputs = indexer_kv_inputs
        self.indexer_state_inputs = indexer_state_inputs
        self.page_size = int(page_size)
        self.is_cuda_graph = bool(is_cuda_graph)

    @property
    def requires_cuda_graph_prepare(self) -> bool:
        return not self.is_cuda_graph

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs_or_by_tag: Any) -> None:
        self.full_inputs = _select_group_for_tag(attn_inputs_or_by_tag, FULL_ATTN_KV_TAG)
        self.indexer_kv_inputs = _select_group_for_tag(attn_inputs_or_by_tag, INDEXER_KV_TAG)
        self.indexer_state_inputs = _select_group_for_tag(attn_inputs_or_by_tag, INDEXER_STATE_TAG)

    def forward_dense_step(
        self,
        q_3d: torch.Tensor,
        kv_cache_layer: Any,
        *,
        bt_kv: torch.Tensor,
        kv_lens: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        return _qwen4_5_paged_causal_step_attn_gpu(
            q_3d,
            kv_cache_layer,
            bt_kv,
            kv_lens,
            active_rows,
            kv_heads=int(self.arch_cfg.num_key_value_heads),
            head_dim=int(self.arch_cfg.head_dim),
        )

    def forward_qsa_step(
        self,
        q_3d: torch.Tensor,
        kv_cache_layer: Any,
        *,
        bt_kv: torch.Tensor,
        kv_lens: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
            qsa_sparse_paged_decode_attention,
        )

        out = qsa_sparse_paged_decode_attention(
            q_3d.unsqueeze(2),
            kv_cache_layer.kv_cache_base,
            bt_kv,
            kv_lens.unsqueeze(1) if kv_lens.ndim == 1 else kv_lens,
            selected,
            page_size=self.page_size,
            kv_heads=int(self.arch_cfg.num_key_value_heads),
            head_dim=int(self.arch_cfg.head_dim),
            kv_scale=getattr(kv_cache_layer, "kv_scale_base", None),
            output_gate=None,
            is_cuda_graph=self.is_cuda_graph,
        )
        return out.squeeze(2)

    def forward_qsa_batch(
        self,
        q_bhsd: torch.Tensor,
        kv_cache_layer: Any,
        *,
        bt_kv: torch.Tensor,
        kv_lens_2d: torch.Tensor,
        selected_3d: torch.Tensor,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
            qsa_sparse_paged_decode_attention,
        )

        return qsa_sparse_paged_decode_attention(
            q_bhsd,
            kv_cache_layer.kv_cache_base,
            bt_kv,
            kv_lens_2d,
            selected_3d,
            page_size=self.page_size,
            kv_heads=int(self.arch_cfg.num_key_value_heads),
            head_dim=int(self.arch_cfg.head_dim),
            kv_scale=getattr(kv_cache_layer, "kv_scale_base", None),
            output_gate=output_gate,
            is_cuda_graph=self.is_cuda_graph,
        )


class Qwen4_5DecodeAttnHandle:
    """Persistent FlashInfer paged-decode handle compatible with ``CudaGraphRunner``."""

    def __init__(
        self,
        attn_configs: Any,
        full_attn_inputs: Any,
        *,
        is_cuda_graph: bool = False,
    ) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeAttnOp,
        )

        self.is_cuda_graph = bool(is_cuda_graph)
        if bool(getattr(full_attn_inputs, "is_cuda_graph", False)) != self.is_cuda_graph:
            try:
                full_attn_inputs.is_cuda_graph = self.is_cuda_graph
            except Exception:
                pass
        self.op = PyFlashinferDecodeAttnOp(attn_configs, full_attn_inputs)
        self.op.enable_cuda_graph = self.is_cuda_graph
        self.params = self.op.prepare(full_attn_inputs)

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs_or_by_tag: Any) -> None:
        full_inputs = _select_group_for_tag(attn_inputs_or_by_tag, FULL_ATTN_KV_TAG)
        self.op.prepare_for_cuda_graph_replay(full_inputs)

    def forward(self, q_3d: torch.Tensor, kv_cache_layer: Any) -> torch.Tensor:
        return self.op.forward(q_3d, kv_cache_layer, self.params)


class Qwen4_5QSADecodeHandle:
    """Zero-sync QSA paged-decode handle compatible with ``CudaGraphRunner``."""

    def __init__(
        self,
        arch_cfg: Qwen4_5ArchConfig,
        full_attn_inputs: Any,
        indexer_kv_inputs: Any = None,
        indexer_state_inputs: Any = None,
        *,
        page_size: int,
        is_cuda_graph: bool = False,
    ) -> None:
        self.arch_cfg = arch_cfg
        self.full_inputs = full_attn_inputs
        self.indexer_kv_inputs = indexer_kv_inputs
        self.indexer_state_inputs = indexer_state_inputs
        self.page_size = int(page_size)
        self.is_cuda_graph = bool(is_cuda_graph)

    @property
    def requires_cuda_graph_prepare(self) -> bool:
        return not self.is_cuda_graph

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs_or_by_tag: Any) -> None:
        self.full_inputs = _select_group_for_tag(attn_inputs_or_by_tag, FULL_ATTN_KV_TAG)
        self.indexer_kv_inputs = _select_group_for_tag(attn_inputs_or_by_tag, INDEXER_KV_TAG)
        self.indexer_state_inputs = _select_group_for_tag(attn_inputs_or_by_tag, INDEXER_STATE_TAG)

    def forward(
        self,
        q_3d: torch.Tensor,
        kv_cache_layer: Any,
        *,
        bt_kv: torch.Tensor,
        kv_lens: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
            qsa_sparse_paged_decode_attention,
        )

        out = qsa_sparse_paged_decode_attention(
            q_3d.unsqueeze(2),
            kv_cache_layer.kv_cache_base,
            bt_kv,
            kv_lens.unsqueeze(1) if kv_lens.ndim == 1 else kv_lens,
            selected,
            page_size=self.page_size,
            kv_heads=int(self.arch_cfg.num_key_value_heads),
            head_dim=int(self.arch_cfg.head_dim),
            kv_scale=getattr(kv_cache_layer, "kv_scale_base", None),
            output_gate=None,
            is_cuda_graph=self.is_cuda_graph,
        )
        return out.squeeze(2)


class Qwen4_5PrefillHandle:
    """Zero-sync static-metadata handle enabling short-sequence Prefill CUDA Graph capture."""

    def __init__(
        self,
        arch_cfg: Qwen4_5ArchConfig,
        full_attn_inputs: Any,
        *,
        in_lens: Tuple[int, ...],
        pfx_lens: Tuple[int, ...],
        plan: "_RaggedPrefillPlan",
        flat_pos: Optional[torch.Tensor],
        is_cuda_graph: bool = False,
        is_auto_cg_dynamic_pfx: bool = False,
        static_vis_q: Optional[torch.Tensor] = None,
        cg_max_ctx_tokens: Optional[int] = None,
        sp_sync_all_blocks_override: Optional[bool] = None,
    ) -> None:
        self.arch_cfg = arch_cfg
        self.full_inputs = full_attn_inputs
        self.in_lens = tuple(int(x) for x in in_lens)
        self.pfx_lens = tuple(int(x) for x in pfx_lens)
        self.plan = plan
        self.flat_pos = flat_pos
        self.is_cuda_graph = bool(is_cuda_graph)
        self.is_auto_cg_dynamic_pfx = bool(is_auto_cg_dynamic_pfx)
        self.static_vis_q = static_vis_q
        self.cg_max_ctx_tokens = cg_max_ctx_tokens
        self.sp_sync_all_blocks_override = sp_sync_all_blocks_override
        self.init_bufs: Dict[Any, torch.Tensor] = {}
        self.snap_phys_bufs: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}
        self.sp_snap_phys_bufs: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}

    @property
    def requires_cuda_graph_prepare(self) -> bool:
        return not self.is_cuda_graph

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs_or_by_tag: Any) -> None:
        self.full_inputs = _select_group_for_tag(attn_inputs_or_by_tag, FULL_ATTN_KV_TAG)


def _model_a2a_step_scope(model: Any) -> Any:
    """Return explicit per-step equal_tokens cache scope when the model carries A2A comm."""
    a2a = getattr(model, "all_to_all_comm", None)
    if a2a is None:
        for lyr in getattr(model, "layers", ()):
            mlp = getattr(lyr, "mlp", None)
            a2a = getattr(mlp, "all_to_all_comm", None) if mlp is not None else None
            if a2a is not None:
                break
    scope_fn = getattr(a2a, "step_equal_tokens_scope", None) if a2a is not None else None
    return scope_fn() if callable(scope_fn) else nullcontext()


def _run_runtime_mlp(
    layer: Any,
    mlp_in: torch.Tensor,
    *,
    sp_input: bool = False,
    sp_total_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch layer MLP with optional SP sharding or 3-stage EP TBO (QWEN4_5_RUNTIME_MOE_TBO=1)."""
    if sp_input:
        return layer.mlp(mlp_in, sp_input=True, sp_total_tokens=sp_total_tokens)
    if (
        mlp_in.shape[0] >= 2
        and os.environ.get("QWEN4_5_RUNTIME_MOE_TBO", "0") == "1"
        and hasattr(layer, "forward_mlp_tbo")
    ):
        return layer.forward_mlp_tbo(
            mlp_in,
            use_reference=not bool(getattr(layer.mlp, "use_triton_mxfp4", True)),
        )
    return layer.mlp(mlp_in)


@dataclass
class _RaggedSPSubPlan:
    """Local SP shard sub-plan covering global tokens ``[s0, s1)``."""

    sp_size: int
    sp_rank: int
    s0: int
    s1: int
    n_loc: int
    sub_bsz: int
    sub_reqs: List[int]
    sub_lens: List[int]
    sub_pfx: List[int]
    sub_cu_host: List[int]
    loc_start: int
    req0_start: int
    has_straddle: bool
    cu_seqlens_i32: torch.Tensor
    tok_req_i32: torch.Tensor
    tile_req_16: torch.Tensor
    tile_t0_16: torch.Tensor
    tile_req_32: torch.Tensor
    tile_t0_32: torch.Tensor
    sub_req_idx_dev: torch.Tensor
    sub_reqs_contiguous: bool = False
    halo_bounds: List[Tuple[int, int]] = field(default_factory=list)
    snap_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]] = field(
        default_factory=dict
    )
    snap_slot_cache: Dict[int, Tuple[Optional[Tuple[int, int]], torch.Tensor]] = field(
        default_factory=dict
    )


@dataclass
class _RaggedPrefillPlan:
    """Precomputed device metadata for whole-batch packed ragged prefill."""

    bsz: int
    total_tokens: int
    active_reqs: List[int]
    active_lens: List[int]
    active_pfx: List[int]
    offsets: List[int]
    cu_seqlens_host: List[int]
    has_any_prefix: bool
    is_uniform_batch: bool
    cu_seqlens_i32: torch.Tensor
    start_pos_i64: torch.Tensor
    prefix_i32: torch.Tensor
    tok_req_i32: torch.Tensor
    tok_abs_pos_i32: torch.Tensor
    tok_abs_pos_i64: torch.Tensor
    tile_req_16: torch.Tensor
    tile_t0_16: torch.Tensor
    tile_req_32: torch.Tensor
    tile_t0_32: torch.Tensor
    chunk_req_64: torch.Tensor
    chunk_t0_64: torch.Tensor
    req_chunk_start_64: torch.Tensor
    act_idx_dev: Optional[torch.Tensor]
    snap_host_cache: Dict[int, Tuple[List[int], List[int], List[int], List[int], List[int], bool, bool]] = field(
        default_factory=dict
    )
    snap_cache: Dict[
        int,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            bool,
            bool,
        ],
    ] = field(default_factory=dict)
    sp_subplan_cache: Dict[Tuple[int, int], _RaggedSPSubPlan] = field(default_factory=dict)
    sp_sync_cache: Dict[Tuple[int, bool, int], Tuple[List[Optional[torch.Tensor]], List[int]]] = field(
        default_factory=dict
    )
    sp_sync_host_cache: Dict[Tuple[int, bool, int], List[List[int]]] = field(
        default_factory=dict
    )
    snap_rel_pages_cache: Dict[int, torch.Tensor] = field(default_factory=dict)
    sp_prefill_sync_meta_cache: Dict[Any, Any] = field(default_factory=dict)


class Qwen4_5PagedStateBridge:
    """Adapts RTP-LLM paged ``KVCache`` pools <-> ``Qwen4_5DecoderLayer`` execution."""

    def __init__(
        self,
        arch_cfg: Qwen4_5ArchConfig,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_size: int = 1,
        dp_rank: int = 0,
        dp_group: Any = None,
        dp_attn: bool = False,
        fast_all_reduce: Any = None,
        sp_fast_comm: Any = None,
    ) -> None:
        self.cfg = arch_cfg
        self.tp_size = max(1, int(tp_size))
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_size = max(1, int(dp_size))
        self.dp_rank = int(dp_rank)
        self.dp_group = dp_group
        self.dp_attn = bool(dp_attn)
        self.fast_all_reduce = fast_all_reduce
        self.sp_fast_comm = (
            sp_fast_comm
            if (
                sp_fast_comm is not None
                and self.sp_group is not None
                and getattr(sp_fast_comm, "group", None) is self.sp_group
            )
            else (
                fast_all_reduce
                if (
                    fast_all_reduce is not None
                    and self.sp_group is not None
                    and getattr(fast_all_reduce, "group", None) is self.sp_group
                )
                else None
            )
        )
        self._ragged_plan_cache: "OrderedDict[Any, _RaggedPrefillPlan]" = OrderedDict()
        self._prefill_init_scratch: Dict[Any, torch.Tensor] = {}
        self._dec_sp_meta_cache: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
        local_k_heads = arch_cfg.linear_num_key_heads // self.tp_size
        local_v_heads = arch_cfg.linear_num_value_heads // self.tp_size
        self.ssm_shape = (
            local_v_heads,
            arch_cfg.linear_key_head_dim,
            arch_cfg.linear_value_head_dim,
        )
        k_dim = local_k_heads * arch_cfg.linear_key_head_dim
        v_dim = local_v_heads * arch_cfg.linear_value_head_dim
        k_conv = max(0, arch_cfg.linear_conv_kernel_dim - 1)
        self.gdn_conv_shape = (2 * k_dim + v_dim, k_conv)
        self.gdn_consumer_conv_shape = (k_dim, k_conv)
        full_kv_dim = arch_cfg.num_key_value_heads * arch_cfg.head_dim
        self._opaque_shapes: Dict[str, Tuple[int, ...]] = {
            GDN_SSM_STATE_TAG: self.ssm_shape,
            GDN_CONV_STATE_TAG: self.gdn_conv_shape,
            GDN_CONSUMER_Q_CONV_TAG: self.gdn_consumer_conv_shape,
            FULL_ATTN_KV_SCONV_TAG: (2, full_kv_dim, k_conv),
            FULL_ATTN_OUTPUT_SCONV_TAG: (arch_cfg.hidden_size, k_conv),
            LINEAR_ATTN_OUTPUT_SCONV_TAG: (arch_cfg.hidden_size, k_conv),
            FFN_OUTPUT_SCONV_TAG: (arch_cfg.hidden_size, k_conv),
            PLE_STATE_TAG: (
                arch_cfg.hc_count * arch_cfg.hidden_size,
                max(0, arch_cfg.ple_conv_kernel_size - 1),
            ),
            PLE_NGRAM_CTX_TAG: (max(0, arch_cfg.ngram_size - 1),),
            INDEXER_STATE_TAG: (
                2 * max(1, int(arch_cfg.indexer_compress_ratio)),
                max(1, int(arch_cfg.indexer_head_dim)),
            ),
        }

    def _cp_indexer_kwargs(self) -> Dict[str, Any]:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return {}
        if self.sp_size > 1 and self.sp_group is not None and (not self.dp_attn or self.tp_size >= self.sp_size):
            return {
                "sp_size": self.sp_size,
                "sp_rank": self.sp_rank,
                "sp_group": self.sp_group,
                "fast_all_reduce": self.fast_all_reduce,
            }
        if self.dp_attn:
            return {}
        if self.tp_size > 1 and self.tp_group is not None:
            return {
                "sp_size": self.tp_size,
                "sp_rank": self.tp_rank,
                "sp_group": self.tp_group,
                "fast_all_reduce": self.fast_all_reduce,
            }
        return {}

    def _opaque_target_spec(
        self, tag: str, base: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tuple[int, ...], torch.dtype]:
        if tag == PLE_NGRAM_CTX_TAG:
            return self._opaque_shapes[tag], torch.int64
        if tag in (GDN_SSM_STATE_TAG, INDEXER_STATE_TAG):
            return self._opaque_shapes[tag], torch.float32
        target_shape = self._opaque_shapes.get(tag, tuple(int(d) for d in base.shape[1:]))
        target_dtype = dtype if dtype is not None else base.dtype
        if target_dtype == torch.uint8:
            target_dtype = torch.bfloat16
        return target_shape, target_dtype

    def _unpack_decode_kv_to_padded(
        self,
        cache: Any,
        block_table: torch.Tensor,
        prev_lens: List[int],
        max_prev: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        allow_zero_padded: bool = False,
    ) -> FullAttnKVCache:
        bsz = len(prev_lens)
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        h_kv, h_d = self.cfg.num_key_value_heads, self.cfg.head_dim
        k_buf = torch.zeros((bsz, max_prev + 1, h_kv, h_d), dtype=dtype, device=device)
        v_buf = torch.zeros((bsz, max_prev + 1, h_kv, h_d), dtype=dtype, device=device)
        for b, plen in enumerate(prev_lens):
            if plen > 0:
                k_req, v_req = _load_single_req_full_attn_kv(
                    base,
                    block_table[b],
                    plen,
                    tpb,
                    num_kv_heads=h_kv,
                    head_dim=h_d,
                    allow_zero_padded=allow_zero_padded,
                    kv_scale=getattr(cache, "kv_scale_base", None),
                    out_dtype=dtype,
                )
                k_buf[b, :plen].copy_(k_req)
                v_buf[b, :plen].copy_(v_req)
        return FullAttnKVCache(k_cache=k_buf, v_cache=v_buf, seq_len=max_prev)

    def _read_opaque_batch(
        self,
        cache: Any,
        block_table: torch.Tensor,
        prev_lens: List[int],
        tag: str,
        layer_idx: int,
        *,
        fill_value: int = 0,
        target_dtype: Optional[torch.dtype] = None,
        allow_zero_padded: bool = False,
    ) -> torch.Tensor:
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        bsz = len(prev_lens)
        target_shape, resolved_dtype = self._opaque_target_spec(tag, base, target_dtype)
        if fill_value == 0:
            out = torch.zeros((bsz, *target_shape), dtype=resolved_dtype, device=base.device)
        else:
            out = torch.full((bsz, *target_shape), fill_value, dtype=resolved_dtype, device=base.device)
        for b, plen in enumerate(prev_lens):
            if plen > 0:
                page = (plen - 1) // tpb
                raw_phys = int(block_table[b, page].item())
                if allow_zero_padded and raw_phys == 0:
                    continue
                phys = _check_phys_block(raw_phys, int(base.shape[0]), tag, layer_idx)
                out[b].copy_(_view_opaque_row(base, phys, target_shape, resolved_dtype))
        return out

    def _write_opaque_batch(
        self,
        cache: Any,
        block_table: torch.Tensor,
        new_lens: List[int],
        tensor: torch.Tensor,
        tag: str,
        layer_idx: int,
        *,
        allow_zero_padded: bool = False,
    ) -> None:
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        target_shape = tuple(int(d) for d in tensor.shape[1:])
        target_dtype = tensor.dtype
        for b, nlen in enumerate(new_lens):
            if nlen > 0:
                page = (nlen - 1) // tpb
                raw_phys = int(block_table[b, page].item())
                if allow_zero_padded and raw_phys == 0:
                    continue
                phys = _check_phys_block(raw_phys, int(base.shape[0]), tag, layer_idx)
                _view_opaque_row(base, phys, target_shape, target_dtype).copy_(tensor[b])

    @staticmethod
    def _opaque_2d_aligned_view(
        tensor: torch.Tensor,
        rows: int,
        need_bytes: int,
        elem_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return [rows, row_elems] view in ``elem_dtype`` if stride and ptr satisfy alignment."""
        elem_sz = _dtype_itemsize(elem_dtype)
        if need_bytes % elem_sz != 0:
            return None
        t_u8 = tensor.reshape(rows, -1).view(torch.uint8)
        if int(t_u8.stride(0)) % elem_sz != 0 or t_u8.data_ptr() % elem_sz != 0 or int(t_u8.shape[1]) % elem_sz != 0:
            return None
        viewed, _ = _view_opaque_pool_2d(tensor, elem_dtype)
        return viewed if viewed.stride(1) == 1 else None

    def _read_opaque_batch_gpu(
        self,
        cache: Any,
        block_table: torch.Tensor,
        seq_before_dev: torch.Tensor,
        tag: str,
        layer_idx: int,
        *,
        fill_value: int = 0,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        bsz = int(seq_before_dev.shape[0])
        target_shape, resolved_dtype = self._opaque_target_spec(tag, base, target_dtype)
        num_blocks = int(base.shape[0])
        max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
        if bsz == 0 or max_blocks == 0 or num_blocks == 0:
            if fill_value == 0:
                return torch.zeros((bsz, *target_shape), dtype=resolved_dtype, device=base.device)
            return torch.full(
                (bsz, *target_shape), fill_value, dtype=resolved_dtype, device=base.device
            )

        out = torch.empty((bsz, *target_shape), dtype=resolved_dtype, device=base.device)
        return self._read_opaque_batch_static_gpu(
            cache,
            block_table,
            seq_before_dev,
            out,
            tag,
            layer_idx,
            fill_value=fill_value,
        )

    def _read_opaque_batch_static_gpu(
        self,
        cache: Any,
        block_table: torch.Tensor,
        seq_before_dev: torch.Tensor,
        dst_buf: torch.Tensor,
        tag: str,
        layer_idx: int,
        *,
        fill_value: int = 0,
    ) -> torch.Tensor:
        """Zero-alloc Triton gather of opaque prefix state into preallocated ``dst_buf``."""
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        bsz = int(seq_before_dev.shape[0])
        num_blocks = int(base.shape[0])
        max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
        if bsz == 0 or max_blocks == 0 or num_blocks == 0:
            dst_buf.fill_(fill_value)
            return dst_buf

        dst_u8 = dst_buf.reshape(bsz, -1).view(torch.uint8)
        need_bytes = int(dst_u8.shape[1])
        base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
        if int(base_u8.shape[1]) < need_bytes:
            raise RuntimeError(
                f"Opaque pool row too small: {base_u8.shape[1]} bytes < {need_bytes} bytes "
                f"for tag={tag} layer={layer_idx}"
            )

        if _HAS_TRITON and base.is_cuda and dst_buf.is_cuda:
            dst_view = None
            src_view = None
            n_elems = 0
            fv = 0
            if (dst_buf.dtype == torch.int64 or fill_value == 0) and need_bytes % 8 == 0:
                d8 = self._opaque_2d_aligned_view(dst_buf, bsz, need_bytes, torch.int64)
                s8 = self._opaque_2d_aligned_view(base, num_blocks, need_bytes, torch.int64)
                if d8 is not None and s8 is not None:
                    dst_view, src_view = d8, s8
                    n_elems = need_bytes // 8
                    fv = int(fill_value)
            if dst_view is None and fill_value == 0 and need_bytes % 4 == 0:
                d4 = self._opaque_2d_aligned_view(dst_buf, bsz, need_bytes, torch.int32)
                s4 = self._opaque_2d_aligned_view(base, num_blocks, need_bytes, torch.int32)
                if d4 is not None and s4 is not None:
                    dst_view, src_view = d4, s4
                    n_elems = need_bytes // 4
                    fv = 0
            if dst_view is None and fill_value == 0:
                dst_view = dst_u8 if dst_u8.stride(1) == 1 else dst_u8.contiguous()
                src_view = base_u8 if base_u8.stride(1) == 1 else torch.as_strided(base_u8, size=base_u8.shape, stride=(int(base_u8.stride(0)), 1))
                n_elems = need_bytes
                fv = 0
            if dst_view is not None and src_view is not None and dst_view.data_ptr() == dst_u8.data_ptr():
                block_sz = 1024
                dst_s0, src_s0 = int(dst_view.stride(0)), int(src_view.stride(0))
                bt_s0, bt_s1 = int(block_table.stride(0)), int(block_table.stride(1))
                ptr_or = (
                    dst_view.data_ptr() | src_view.data_ptr()
                    | block_table.data_ptr() | seq_before_dev.data_ptr()
                )
                ckey = (
                    dst_view.device.index, "rt_read_op", dst_view.dtype, src_view.dtype,
                    block_table.dtype, seq_before_dev.dtype,
                    _triton_int_spec(dst_s0), _triton_int_spec(src_s0),
                    _triton_int_spec(bt_s0), _triton_int_spec(bt_s1),
                    _triton_int_spec(tpb), _triton_int_spec(max_blocks),
                    _triton_int_spec(num_blocks), _triton_int_spec(n_elems), _triton_int_spec(fv),
                    block_sz,
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE, ckey, _read_opaque_rows_triton,
                    bsz, int(triton.cdiv(n_elems, block_sz)), 1, ptr_or,
                    (dst_view, src_view, block_table, seq_before_dev,
                     dst_s0, src_s0, bt_s0, bt_s1, tpb, max_blocks, num_blocks, n_elems, fv),
                    {"BLOCK_SIZE": block_sz},
                    num_warps=4,
                )
                return dst_buf

        # Pure PyTorch fallback for non-CUDA or unaligned non-zero fill_value
        pos = seq_before_dev.to(torch.int64)
        read_page = ((pos - 1).clamp(min=0) // tpb).clamp(max=max_blocks - 1)
        read_phys = block_table.gather(1, read_page.unsqueeze(1)).squeeze(1).to(torch.int64)
        valid = (pos > 0) & (read_phys > 0) & (read_phys < num_blocks)
        safe_phys = read_phys.clamp(min=0, max=num_blocks - 1)
        target_shape = tuple(int(d) for d in dst_buf.shape[1:])
        gathered = (
            base_u8[:, :need_bytes]
            .index_select(0, safe_phys)
            .view(dst_buf.dtype)
            .reshape(bsz, *target_shape)
        )
        valid_bc = valid.view(bsz, *([1] * len(target_shape)))
        dst_buf.copy_(gathered)
        dst_buf.masked_fill_(~valid_bc, fill_value)
        return dst_buf

    @staticmethod
    def _try_view_sconv_opaque_pool_2d(
        cache: Any,
        bt: torch.Tensor,
        verify_w: int,
        kernel_size: int,
        need_elems: int,
        dtype: torch.dtype,
        x_ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return zero-copy 2D opaque pool view if direct verify read is valid."""
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        max_blocks = int(bt.shape[1]) if bt.ndim >= 2 else 0
        if not (
            _HAS_TRITON
            and x_ref.is_cuda
            and base.is_cuda
            and int(kernel_size) == 4
            and 0 < verify_w <= 16
            and x_ref.stride(-1) == 1
            and tpb > 0
            and max_blocks > 0
            and int(base.shape[0]) > 0
        ):
            return None
        base_u8 = base.reshape(int(base.shape[0]), -1).view(torch.uint8)
        elem_sz = x_ref.element_size()
        if (
            int(base_u8.shape[1]) < need_elems * elem_sz
            or int(base_u8.shape[1]) % elem_sz != 0
            or int(base_u8.stride(0)) % elem_sz != 0
            or base_u8.data_ptr() % elem_sz != 0
        ):
            return None
        pool_2d, _ = _view_opaque_pool_2d(base, dtype)
        return pool_2d if pool_2d.stride(1) == 1 else None

    def _run_verify_sconv_opaque_or_static(
        self,
        x: torch.Tensor,
        sconv: Any,
        cache: Any,
        bt: torch.Tensor,
        seq_before_dev: torch.Tensor,
        tx_stage: _PinnedTargetVerifyTxStager,
        tag: str,
        lid: int,
        state_shape: Tuple[int, ...],
        stage_states: torch.Tensor,
        *,
        clamp_val: Optional[float] = None,
    ) -> torch.Tensor:
        """Run K=4 verify conv directly from opaque pool (or fall back to static gather)."""
        verify_w = int(x.shape[1])
        dtype = x.dtype
        pool_2d = self._try_view_sconv_opaque_pool_2d(
            cache, bt, verify_w, sconv.kernel_size, math.prod(state_shape), dtype, x
        )
        if pool_2d is not None:
            return causal_dwconv1d_verify_opaque_triton(
                x,
                sconv.weight,
                pool_2d,
                bt,
                seq_before_dev,
                stage_states,
                page_size=int(cache.seq_size_per_block),
                pool_col_offset=0,
                residual=sconv.residual,
                use_activation=sconv.use_activation,
                clamp_val=clamp_val,
            )
        init_buf = self._read_opaque_batch_static_gpu(
            cache,
            bt,
            seq_before_dev,
            tx_stage.scratch_tensor(("init", tag, lid), (int(x.shape[0]), *state_shape), dtype, x.device),
            tag,
            lid,
        )
        return causal_dwconv1d_verify_triton(
            x,
            sconv.weight,
            init_buf,
            stage_states,
            residual=sconv.residual,
            use_activation=sconv.use_activation,
            clamp_val=clamp_val,
        )

    def _resolve_sp_dist(self) -> Tuple[int, int, Any, Any]:
        if (
            self.sp_size <= 1
            or self.sp_group is None
            or (self.dp_attn and self.tp_size < self.sp_size)
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return 1, 0, None, None
        return self.sp_size, self.sp_rank, self.sp_group, self.sp_fast_comm

    @staticmethod
    def _sp_all_gather_dim0(
        x_loc: torch.Tensor,
        *,
        total_tokens: int,
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any = None,
    ) -> torch.Tensor:
        if total_tokens == 0:
            return torch.empty((0, *x_loc.shape[1:]), dtype=x_loc.dtype, device=x_loc.device)
        if sp_comm is not None and hasattr(sp_comm, "all_gather_sp"):
            return sp_comm.all_gather_sp(x_loc, total_tokens=total_tokens)
        chunk = (total_tokens + sp_size - 1) // sp_size
        padded_t = sp_size * chunk
        rest = tuple(x_loc.shape[1:])
        x_c = x_loc if x_loc.is_contiguous() else x_loc.contiguous()
        if int(x_c.shape[0]) == chunk:
            src_pad = x_c
        else:
            src_pad = torch.empty((chunk, *rest), dtype=x_c.dtype, device=x_c.device)
            copy_n = min(int(x_c.shape[0]), chunk)
            if copy_n > 0:
                src_pad[:copy_n].copy_(x_c[:copy_n])
        if not x_c.is_cuda or "gloo" in str(dist.get_backend(sp_group)).lower():
            cpu_in = src_pad.detach().cpu() if src_pad.is_cuda else src_pad
            chunks = [torch.empty_like(cpu_in) for _ in range(sp_size)]
            dist.all_gather(chunks, cpu_in, group=sp_group)
            full_cpu = torch.cat(chunks, dim=0)[:total_tokens]
            return full_cpu.to(device=x_c.device, dtype=x_c.dtype) if x_c.is_cuda else full_cpu
        dst_full = torch.empty((padded_t, *rest), dtype=x_c.dtype, device=x_c.device)
        dist.all_gather_into_tensor(dst_full.view(sp_size, chunk, *rest), src_pad, group=sp_group)
        return dst_full[:total_tokens]

    @staticmethod
    def _sp_reduce_scatter_dim0(
        x_full: torch.Tensor,
        *,
        total_tokens: int,
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any = None,
    ) -> torch.Tensor:
        if total_tokens == 0:
            return torch.empty((0, *x_full.shape[1:]), dtype=x_full.dtype, device=x_full.device)
        if sp_comm is not None and hasattr(sp_comm, "reduce_scatter_sp"):
            return sp_comm.reduce_scatter_sp(x_full, total_tokens=total_tokens)
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        s0, s1 = sp_slice_bounds(total_tokens, sp_size, sp_rank)
        chunk = (total_tokens + sp_size - 1) // sp_size
        padded_t = sp_size * chunk
        rest = tuple(x_full.shape[1:])
        x_c = x_full if x_full.is_contiguous() else x_full.contiguous()
        if int(x_c.shape[0]) == padded_t:
            src_pad = x_c
        else:
            src_pad = torch.zeros((padded_t, *rest), dtype=x_c.dtype, device=x_c.device)
            src_pad[:total_tokens].copy_(x_c[:total_tokens])
        if not x_c.is_cuda or "gloo" in str(dist.get_backend(sp_group)).lower():
            cpu_in = (src_pad.detach().cpu() if src_pad.is_cuda else src_pad).clone()
            dist.all_reduce(cpu_in, op=dist.ReduceOp.SUM, group=sp_group)
            out_cpu = cpu_in[s0:s1]
            return out_cpu.to(device=x_c.device, dtype=x_c.dtype) if x_c.is_cuda else out_cpu.clone()
        dst_chunk = torch.empty((chunk, *rest), dtype=x_c.dtype, device=x_c.device)
        dist.reduce_scatter_tensor(dst_chunk, src_pad, op=dist.ReduceOp.SUM, group=sp_group)
        return dst_chunk[: s1 - s0]

    @staticmethod
    def _hc_mix_from_normed_safe(hc_mod: Any, normed: torch.Tensor, h: torch.Tensor, prev_latent: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if h.shape[0] == 0:
            return (
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                h,
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                prev_latent,
            )
        return hc_mod.mix_from_normed(normed, h, prev_latent)

    @staticmethod
    def _hc_mix_with_latent_safe(hc_mod: Any, h: torch.Tensor, prev_latent: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if h.shape[0] == 0:
            return (
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                h.repeat(1, hc_mod.hc_count) if int(h.shape[-1]) == hc_mod.hidden_size else h,
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                prev_latent,
            )
        return hc_mod.mix_with_latent(h, prev_latent)

    @staticmethod
    def _hc_combine_and_mix_safe(
        hc_mod: Any, sub_out: torch.Tensor, h: torch.Tensor, inj: torch.Tensor, prev_latent: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if h.shape[0] == 0:
            return (
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                h,
                torch.empty((0, hc_mod.hidden_size), dtype=h.dtype, device=h.device),
                prev_latent,
            )
        return hc_mod.combine_and_mix_with_latent(sub_out, h, inj, prev_latent)

    @staticmethod
    def _hc_combine_safe(hc_mod: Any, sub_out: torch.Tensor, h: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        if h.shape[0] == 0:
            return h
        return hc_mod.combine(sub_out, h, inj)

    def _run_empty_ep_collectives(
        self,
        model: Qwen4_5Model,
        dtype: torch.dtype,
        device: torch.device,
        *,
        sp_active: Optional[bool] = None,
    ) -> None:
        """Keep ``DP>1, EP>1`` empty-batch ranks participating in MoE AllToAll collectives."""
        if (
            self.dp_size <= 1
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        eff_sp, _, _, _ = self._resolve_sp_dist()
        use_sp = bool(sp_active) if sp_active is not None else (eff_sp > 1)
        empty_in = torch.empty((0, self.cfg.hidden_size), dtype=dtype, device=device)
        with _model_a2a_step_scope(model):
            for layer in model.layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None and (
                    int(getattr(mlp, "ep_size", 1) or 1) > 1
                    or getattr(mlp, "all_to_all_comm", None) is not None
                ):
                    _ = _run_runtime_mlp(
                        layer,
                        empty_in,
                        sp_input=use_sp,
                        sp_total_tokens=(0 if use_sp else None),
                    )

    def _build_ragged_prefill_plan(
        self,
        input_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
        device: torch.device,
        *,
        freeze: bool = False,
    ) -> _RaggedPrefillPlan:
        in_key = tuple(int(x) for x in input_lengths)
        pfx_key = tuple(int(x) for x in prefix_lengths)
        cache_key = (device.type, device.index if device.index is not None else -1, in_key, pfx_key)
        if not freeze:
            hit = self._ragged_plan_cache.get(cache_key)
            if hit is not None:
                self._ragged_plan_cache.move_to_end(cache_key)
                return hit

        bsz = len(in_key)
        active_reqs: List[int] = []
        active_lens: List[int] = []
        active_pfx: List[int] = []
        offsets: List[int] = []
        cu_list: List[int] = [0]
        cur = 0
        for b in range(bsz):
            ln = in_key[b]
            if ln > 0:
                active_reqs.append(b)
                active_lens.append(ln)
                active_pfx.append(pfx_key[b])
                offsets.append(cur)
                cur += ln
                cu_list.append(cur)

        b_act = len(active_reqs)
        t_req_list: List[int] = []
        t_abs_list: List[int] = []
        tr16: List[int] = []
        tt16: List[int] = []
        tr32: List[int] = []
        tt32: List[int] = []
        cr64: List[int] = []
        ct64: List[int] = []
        rc_start: List[int] = [0]

        for bi in range(b_act):
            ln = active_lens[bi]
            pf = active_pfx[bi]
            t_req_list.extend([bi] * ln)
            t_abs_list.extend(range(pf, pf + ln))
            for t0 in range(0, ln, 16):
                tr16.append(bi)
                tt16.append(t0)
            for t0 in range(0, ln, 32):
                tr32.append(bi)
                tt32.append(t0)
            for t0 in range(0, ln, 64):
                cr64.append(bi)
                ct64.append(t0)
            rc_start.append(len(cr64))

        need_act_idx = not (b_act == bsz and active_reqs == list(range(bsz)))
        segs: List[List[int]] = [
            cu_list,
            active_pfx,
            t_req_list,
            t_abs_list,
            tr16,
            tt16,
            tr32,
            tt32,
            cr64,
            ct64,
            rc_start,
        ]
        if need_act_idx:
            segs.append(active_reqs)
        flat_host: List[int] = []
        spans: List[Tuple[int, int]] = []
        for lst in segs:
            s0 = len(flat_host)
            ln_s = len(lst)
            flat_host.extend(lst)
            pad = (-ln_s) & 3
            if pad:
                flat_host.extend([0] * pad)
            spans.append((s0, s0 + ln_s))
        packed_dev = _copy_i32_list_to_dev(flat_host, device)
        views = [packed_dev[a:b] for (a, b) in spans]
        cu_dev, pfx_dev, treq_dev, tabs_dev, tr16_d, tt16_d, tr32_d, tt32_d, cr64_d, ct64_d, rc64_d = views[:11]
        act_idx_dev = views[11].to(torch.int64) if need_act_idx else None
        is_uniform = (
            b_act > 0
            and len(set(active_lens)) == 1
            and len(set(active_pfx)) == 1
            and active_lens[0] > 0
        )
        plan_obj = _RaggedPrefillPlan(
            bsz=b_act,
            total_tokens=cur,
            active_reqs=active_reqs,
            active_lens=active_lens,
            active_pfx=active_pfx,
            offsets=offsets,
            cu_seqlens_host=cu_list,
            has_any_prefix=any(p > 0 for p in active_pfx),
            is_uniform_batch=is_uniform,
            cu_seqlens_i32=cu_dev,
            start_pos_i64=pfx_dev.to(torch.int64),
            prefix_i32=pfx_dev,
            tok_req_i32=treq_dev,
            tok_abs_pos_i32=tabs_dev,
            tok_abs_pos_i64=tabs_dev.to(torch.int64),
            tile_req_16=tr16_d,
            tile_t0_16=tt16_d,
            tile_req_32=tr32_d,
            tile_t0_32=tt32_d,
            chunk_req_64=cr64_d,
            chunk_t0_64=ct64_d,
            req_chunk_start_64=rc64_d,
            act_idx_dev=act_idx_dev,
        )
        if not freeze and not (device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            self._ragged_plan_cache[cache_key] = plan_obj
            if len(self._ragged_plan_cache) > 64:
                self._ragged_plan_cache.popitem(last=False)
        return plan_obj

    @staticmethod
    def _compute_snap_host(
        lens: Sequence[int],
        pfx: Sequence[int],
        offsets: Sequence[int],
        tpb: int,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], bool, bool]:
        snap_toks: List[int] = []
        snap_reqs: List[int] = []
        snap_pages: List[int] = []
        snap_local_ends: List[int] = []
        req_snap_start: List[int] = [0]
        aligned_16 = True
        aligned_32 = True
        for bi in range(len(lens)):
            t_in = lens[bi]
            t_pfx = pfx[bi]
            off_b = offsets[bi]
            pos_c = t_pfx
            rem = t_in
            loc_c = 0
            while rem > 0:
                if tpb > 0:
                    next_bound = ((pos_c // tpb) + 1) * tpb
                    take = min(rem, next_bound - pos_c)
                    pg = pos_c // tpb
                else:
                    take = rem
                    pg = 0
                loc_end = loc_c + take
                if loc_end != t_in:
                    if (loc_end & 15) != 0:
                        aligned_16 = False
                    if (loc_end & 31) != 0:
                        aligned_32 = False
                snap_toks.append(off_b + loc_end - 1)
                snap_reqs.append(bi)
                snap_pages.append(pg)
                snap_local_ends.append(loc_end)
                pos_c += take
                loc_c = loc_end
                rem -= take
            req_snap_start.append(len(snap_toks))
        return snap_toks, snap_reqs, snap_pages, snap_local_ends, req_snap_start, aligned_16, aligned_32

    @classmethod
    def _get_snap_schedule(
        cls, plan: _RaggedPrefillPlan, tpb: int, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        bool,
    ]:
        cached = plan.snap_cache.get(tpb)
        if cached is not None:
            return cached
        snap_toks, snap_reqs, snap_pages, snap_local_ends, req_snap_start, aligned_16, aligned_32 = (
            cls._compute_snap_host(plan.active_lens, plan.active_pfx, plan.offsets, tpb)
        )
        plan.snap_host_cache[tpb] = (
            snap_toks, snap_reqs, snap_pages, snap_local_ends, req_snap_start, aligned_16, aligned_32
        )
        flat_host: List[int] = []
        spans: List[Tuple[int, int]] = []
        for lst in (snap_toks, snap_reqs, snap_pages, snap_local_ends, req_snap_start):
            s0 = len(flat_host)
            ln_s = len(lst)
            flat_host.extend(lst)
            pad = (-ln_s) & 3
            if pad:
                flat_host.extend([0] * pad)
            spans.append((s0, s0 + ln_s))
        packed_dev = _copy_i32_list_to_dev(flat_host, device)
        s_toks, s_reqs, s_pages_i32, s_ends, r_sstart = [packed_dev[a:b] for (a, b) in spans]
        res = (
            s_toks,
            s_reqs,
            s_pages_i32.to(torch.int64),
            s_ends,
            r_sstart,
            s_reqs.to(torch.int64),
            aligned_16,
            aligned_32,
        )
        plan.snap_cache[tpb] = res
        return res

    @classmethod
    def _get_snap_rel_pages(
        cls, plan: _RaggedPrefillPlan, tpb: int, device: torch.device
    ) -> torch.Tensor:
        cached = plan.snap_rel_pages_cache.get(tpb)
        if cached is not None:
            return cached
        _ = cls._get_snap_schedule(plan, tpb, device)
        _, s_reqs_h, s_pages_h, _, _, _, _ = plan.snap_host_cache[tpb]
        if tpb > 0:
            rel_h = [pg - (plan.active_pfx[rq] // tpb) for pg, rq in zip(s_pages_h, s_reqs_h)]
        else:
            rel_h = list(s_pages_h)
        rel_dev = _copy_i32_list_to_dev(rel_h, device).to(torch.int64)
        plan.snap_rel_pages_cache[tpb] = rel_dev
        return rel_dev

    @staticmethod
    def _build_ragged_sp_subplan(
        plan: _RaggedPrefillPlan,
        sp_size: int,
        sp_rank: int,
        device: torch.device,
    ) -> _RaggedSPSubPlan:
        key = (sp_size, sp_rank)
        cached = plan.sp_subplan_cache.get(key)
        if cached is not None:
            return cached
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        s0, s1 = sp_slice_bounds(plan.total_tokens, sp_size, sp_rank)
        cu_g = plan.cu_seqlens_host
        has_straddle = False
        for rk_i in range(sp_size):
            r0, r1 = sp_slice_bounds(plan.total_tokens, sp_size, rk_i)
            if r1 > r0:
                for gi in range(plan.bsz):
                    if cu_g[gi] < r0 < cu_g[gi + 1]:
                        has_straddle = True
                        break
            if has_straddle:
                break

        sub_reqs: List[int] = []
        sub_lens: List[int] = []
        sub_pfx: List[int] = []
        sub_cu: List[int] = [0]
        sub_treq: List[int] = []
        sub_tr16: List[int] = []
        sub_tt16: List[int] = []
        sub_tr32: List[int] = []
        sub_tt32: List[int] = []
        loc_start = 0
        req0_start = 0
        cur_loc = 0
        for bi in range(plan.bsz):
            rs, re = cu_g[bi], cu_g[bi + 1]
            ov_s = max(s0, rs)
            ov_e = min(s1, re)
            if ov_e <= ov_s:
                continue
            sub_bi = len(sub_reqs)
            if sub_bi == 0:
                loc_start = ov_s - rs
                req0_start = rs
            sub_reqs.append(bi)
            sub_ln = ov_e - ov_s
            sub_lens.append(sub_ln)
            sub_pfx.append(plan.active_pfx[bi] + (ov_s - rs))
            sub_treq.extend([sub_bi] * sub_ln)
            for t0 in range(0, sub_ln, 16):
                sub_tr16.append(sub_bi)
                sub_tt16.append(t0)
            for t0 in range(0, sub_ln, 32):
                sub_tr32.append(sub_bi)
                sub_tt32.append(t0)
            cur_loc += sub_ln
            sub_cu.append(cur_loc)

        segs = [sub_cu, sub_treq, sub_tr16, sub_tt16, sub_tr32, sub_tt32, sub_reqs]
        flat_host: List[int] = []
        spans: List[Tuple[int, int]] = []
        for lst in segs:
            a = len(flat_host)
            flat_host.extend(lst)
            pad = (-len(lst)) & 3
            if pad:
                flat_host.extend([0] * pad)
            spans.append((a, a + len(lst)))
        packed_dev = _copy_i32_list_to_dev(flat_host, device)
        cu_d, treq_d, tr16_d, tt16_d, tr32_d, tt32_d, sub_req_d = [packed_dev[a:b] for (a, b) in spans]
        sub_req_contig = (
            bool(sub_reqs)
            and sub_reqs == list(range(sub_reqs[0], sub_reqs[0] + len(sub_reqs)))
        )
        halo_bounds = (
            [sp_slice_bounds(plan.total_tokens, sp_size, r_i) for r_i in range(sp_size)]
            if has_straddle
            else []
        )
        sub_plan = _RaggedSPSubPlan(
            sp_size=sp_size,
            sp_rank=sp_rank,
            s0=s0,
            s1=s1,
            n_loc=s1 - s0,
            sub_bsz=len(sub_reqs),
            sub_reqs=sub_reqs,
            sub_lens=sub_lens,
            sub_pfx=sub_pfx,
            sub_cu_host=sub_cu,
            loc_start=loc_start,
            req0_start=req0_start,
            has_straddle=has_straddle,
            cu_seqlens_i32=cu_d,
            tok_req_i32=treq_d,
            tile_req_16=tr16_d,
            tile_t0_16=tt16_d,
            tile_req_32=tr32_d,
            tile_t0_32=tt32_d,
            sub_req_idx_dev=sub_req_d.to(torch.int64),
            sub_reqs_contiguous=sub_req_contig,
            halo_bounds=halo_bounds,
        )
        plan.sp_subplan_cache[key] = sub_plan
        return sub_plan

    def _get_sp_snap_base(
        self,
        plan: _RaggedPrefillPlan,
        sub_plan: _RaggedSPSubPlan,
        tpb: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
        cached = sub_plan.snap_cache.get(tpb)
        if cached is not None:
            return cached
        _ = self._get_snap_schedule(plan, tpb, device)
        s_toks_h, s_reqs_h, s_pages_h, _, _, _, _ = plan.snap_host_cache[tpb]
        sub_toks: List[int] = []
        sub_reqs: List[int] = []
        sub_pages: List[int] = []
        sub_ends: List[int] = []
        sub_rss: List[int] = [0]
        aligned_16 = True
        aligned_32 = True
        req_to_sub = {ri: sb for sb, ri in enumerate(sub_plan.sub_reqs)}
        cu_g = plan.cu_seqlens_host
        sub_by_req: Dict[int, List[Tuple[int, int, int, int, int]]] = {ri: [] for ri in sub_plan.sub_reqs}
        for g_slot_i, (g_tok, g_req, g_pg) in enumerate(zip(s_toks_h, s_reqs_h, s_pages_h)):
            if sub_plan.s0 <= g_tok < sub_plan.s1 and g_req in req_to_sub:
                rs = cu_g[g_req]
                loc_off = max(sub_plan.s0, rs) - rs
                loc_end = (g_tok - rs + 1) - loc_off
                sub_bi = req_to_sub[g_req]
                sub_ln = sub_plan.sub_lens[sub_bi]
                if loc_end != sub_ln:
                    if (loc_end & 15) != 0:
                        aligned_16 = False
                    if (loc_end & 31) != 0:
                        aligned_32 = False
                sub_by_req[g_req].append((g_tok - sub_plan.s0, g_req, g_pg, loc_end, g_slot_i))
        g_reqs_list: List[int] = []
        sub_slot_h: List[int] = []
        for sub_bi, g_req in enumerate(sub_plan.sub_reqs):
            for (l_tok, _, g_pg, l_end, g_slot_i) in sub_by_req[g_req]:
                sub_toks.append(l_tok)
                sub_reqs.append(sub_bi)
                g_reqs_list.append(g_req)
                sub_pages.append(g_pg)
                sub_ends.append(l_end)
                sub_slot_h.append(g_slot_i)
            sub_rss.append(len(sub_toks))
        flat_h: List[int] = []
        spans: List[Tuple[int, int]] = []
        for lst in (sub_toks, sub_reqs, g_reqs_list, sub_pages, sub_ends, sub_rss, sub_slot_h):
            a = len(flat_h)
            flat_h.extend(lst)
            pad = (-len(lst)) & 3
            if pad:
                flat_h.extend([0] * pad)
            spans.append((a, a + len(lst)))
        p_dev = _copy_i32_list_to_dev(flat_h, device)
        st_d, sr_d, gr_d, pg_d, se_d, rss_d, slot_d = [p_dev[a:b] for (a, b) in spans]
        slice_ab: Optional[Tuple[int, int]] = (
            (0, 0)
            if not sub_slot_h
            else (
                (sub_slot_h[0], sub_slot_h[0] + len(sub_slot_h))
                if sub_slot_h == list(range(sub_slot_h[0], sub_slot_h[0] + len(sub_slot_h)))
                else None
            )
        )
        sub_plan.snap_slot_cache[tpb] = (slice_ab, slot_d.to(torch.int64))
        res = (
            st_d,
            sr_d,
            gr_d.to(torch.int64),
            pg_d.to(torch.int64),
            se_d,
            rss_d,
            aligned_16,
            aligned_32,
        )
        sub_plan.snap_cache[tpb] = res
        return res

    def _get_sp_snap_schedule(
        self,
        plan: _RaggedPrefillPlan,
        sub_plan: _RaggedSPSubPlan,
        tpb: int,
        bt_act: torch.Tensor,
        device: torch.device,
        *,
        s_phys_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
        st_d, sr_d, gr_i64, pg_i64, se_d, rss_d, aligned_16, aligned_32 = self._get_sp_snap_base(
            plan, sub_plan, tpb, device
        )
        if s_phys_override is not None:
            return st_d, sr_d, se_d, rss_d, s_phys_override, aligned_16, aligned_32
        max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
        if max_cols <= 0 or int(st_d.numel()) == 0:
            s_phys = torch.zeros_like(st_d)
        else:
            s_phys = bt_act[gr_i64, pg_i64.clamp(max=max_cols - 1)].to(torch.int32)
        return st_d, sr_d, se_d, rss_d, s_phys, aligned_16, aligned_32

    def _run_sp_ragged_sconv(
        self,
        x_loc: torch.Tensor,
        sconv: Any,
        plan: _RaggedPrefillPlan,
        sub_plan: _RaggedSPSubPlan,
        init_full: Optional[torch.Tensor],
        cache: Any,
        bt_act: Optional[torch.Tensor],
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any,
        *,
        clamp_val: Optional[float] = None,
        s_phys_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run SP-sharded ShortConv prefill with request-bounded 3-tok halo & snapshot writes."""
        halo_w = max(0, int(sconv.kernel_size) - 1)
        ch = int(x_loc.shape[-1])
        eff_init: Optional[torch.Tensor] = None
        full_from_owner = -1
        bounds = sub_plan.halo_bounds
        if (
            sub_plan.has_straddle
            and halo_w > 0
            and sub_plan.loc_start >= halo_w
            and sub_plan.sub_bsz > 0
            and (sub_plan.s0 - sub_plan.req0_start) >= halo_w
            and bounds
        ):
            for owner_rk, (b0, b1) in enumerate(bounds):
                if b0 <= (sub_plan.s0 - halo_w) and b1 == sub_plan.s0:
                    full_from_owner = owner_rk
                    break
        need_init_base = not (full_from_owner >= 0 and sub_plan.sub_bsz == 1)
        if need_init_base and init_full is not None and sub_plan.sub_bsz > 0:
            if sub_plan.sub_reqs_contiguous:
                r0 = sub_plan.sub_reqs[0]
                r1 = r0 + sub_plan.sub_bsz
                sub_view = init_full if (r0 == 0 and r1 == int(init_full.shape[0])) else init_full[r0:r1]
                eff_init = sub_view.clone() if (sub_plan.loc_start > 0 and full_from_owner < 0) else (
                    torch.empty_like(sub_view) if (sub_plan.loc_start > 0 and sub_plan.sub_bsz == 1) else (
                        sub_view.clone() if sub_plan.loc_start > 0 else sub_view
                    )
                )
            else:
                eff_init = init_full.index_select(0, sub_plan.sub_req_idx_dev)
            if sub_plan.loc_start > 0 and full_from_owner < 0:
                if sub_plan.loc_start >= halo_w:
                    eff_init[0].zero_()
                elif halo_w > 0:
                    keep = halo_w - sub_plan.loc_start
                    eff_init[0, :, :keep] = init_full[sub_plan.sub_reqs[0], :, sub_plan.loc_start :]
                    eff_init[0, :, keep:].zero_()
        if sub_plan.has_straddle and halo_w > 0:
            if sub_plan.n_loc >= halo_w and x_loc.is_contiguous():
                tail_x = x_loc[sub_plan.n_loc - halo_w :]
            else:
                tail_x = torch.zeros((halo_w, ch), dtype=x_loc.dtype, device=x_loc.device)
                take = min(sub_plan.n_loc, halo_w)
                if take > 0:
                    tail_x[halo_w - take :].copy_(x_loc[sub_plan.n_loc - take :])
            gathered = self._sp_all_gather_dim0(
                tail_x,
                total_tokens=sp_size * halo_w,
                sp_size=sp_size,
                sp_rank=sp_rank,
                sp_group=sp_group,
                sp_comm=sp_comm,
            ).view(sp_size, halo_w, ch)
            if sub_plan.loc_start > 0 and sub_plan.sub_bsz > 0:
                if full_from_owner >= 0 and sub_plan.sub_bsz == 1:
                    eff_init = gathered[full_from_owner : full_from_owner + 1].transpose(1, 2)
                elif full_from_owner >= 0:
                    if eff_init is None:
                        eff_init = torch.zeros((sub_plan.sub_bsz, ch, halo_w), dtype=x_loc.dtype, device=x_loc.device)
                    eff_init[0].copy_(gathered[full_from_owner].transpose(0, 1))
                else:
                    if eff_init is None:
                        eff_init = torch.zeros((sub_plan.sub_bsz, ch, halo_w), dtype=x_loc.dtype, device=x_loc.device)
                    for j in range(halo_w):
                        tg = sub_plan.s0 - halo_w + j
                        if tg >= sub_plan.req0_start:
                            for owner_rk, (b0, b1) in enumerate(bounds):
                                if b0 <= tg < b1:
                                    eff_init[0, :, j].copy_(gathered[owner_rk, halo_w - (b1 - tg)])
                                    break
        if sub_plan.n_loc == 0:
            return x_loc
        if cache is not None and bt_act is not None:
            s_tok, s_req, s_end, r_sstart, s_phys, _, a32 = self._get_sp_snap_schedule(
                plan,
                sub_plan,
                int(cache.seq_size_per_block),
                bt_act,
                x_loc.device,
                s_phys_override=s_phys_override,
            )
            pool_bf16, _ = _view_opaque_pool_2d(cache.kv_cache_base, x_loc.dtype)
            return causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                x_loc,
                sconv.weight,
                sub_plan.cu_seqlens_i32,
                sub_plan.tile_req_32,
                sub_plan.tile_t0_32,
                init_state=eff_init,
                snap_tok_idx=s_tok,
                snap_req_idx=s_req,
                dst_pool=pool_bf16,
                snap_dst_idx=s_phys,
                req_snap_start=r_sstart,
                snap_local_ends=s_end,
                snap_aligned=a32,
                skip_invalid_phys=True,
                residual=sconv.residual,
                use_activation=sconv.use_activation,
                clamp_val=clamp_val,
                block_t=32,
            )
        return causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x_loc,
            sconv.weight,
            sub_plan.cu_seqlens_i32,
            sub_plan.tile_req_32,
            sub_plan.tile_t0_32,
            init_state=eff_init,
            residual=sconv.residual,
            use_activation=sconv.use_activation,
            clamp_val=clamp_val,
            block_t=32,
        )

    def _run_sp_step_verify_sconv(
        self,
        x_loc: torch.Tensor,
        sconv: Any,
        cache: Any,
        bt: torch.Tensor,
        seq_before_dev: torch.Tensor,
        tx_stage: "_Qwen4_5OpaqueTxStage",
        tag: str,
        lid: int,
        state_shape: Tuple[int, ...],
        stage_slice: torch.Tensor,
        *,
        v0: int,
        total_v: int,
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any,
        clamp_val: Optional[float] = None,
    ) -> torch.Tensor:
        """Run B=1 Step-SP Verify ShortConv with static-index 3-tok halo exchange."""
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        v_loc = int(x_loc.shape[1])
        ch = int(x_loc.shape[2])
        halo_w = max(0, int(sconv.kernel_size) - 1)
        init_buf = self._read_opaque_batch_static_gpu(
            cache,
            bt,
            seq_before_dev,
            tx_stage.scratch_tensor(("init", tag, lid), (int(x_loc.shape[0]), *state_shape), x_loc.dtype, x_loc.device),
            tag,
            lid,
        )
        eff_init = init_buf
        if halo_w > 0:
            tail_x = torch.zeros((halo_w, ch), dtype=x_loc.dtype, device=x_loc.device)
            take = min(v_loc, halo_w)
            if take > 0:
                tail_x[halo_w - take :].copy_(x_loc[0, v_loc - take :])
            gathered = self._sp_all_gather_dim0(
                tail_x,
                total_tokens=sp_size * halo_w,
                sp_size=sp_size,
                sp_rank=sp_rank,
                sp_group=sp_group,
                sp_comm=sp_comm,
            ).view(sp_size, halo_w, ch)
            if v0 > 0:
                eff_init = tx_stage.scratch_tensor(
                    ("sp_init", tag, lid), (1, *state_shape), x_loc.dtype, x_loc.device
                )
                bounds = [sp_slice_bounds(total_v, sp_size, r) for r in range(sp_size)]
                for j in range(halo_w):
                    tg = v0 - halo_w + j
                    if tg >= 0:
                        for owner_rk, (b0, b1) in enumerate(bounds):
                            if b0 <= tg < b1:
                                eff_init[0, :, j].copy_(gathered[owner_rk, halo_w - (b1 - tg)])
                                break
                    else:
                        eff_init[0, :, j].copy_(init_buf[0, :, halo_w + tg])
        return causal_dwconv1d_verify_triton(
            x_loc,
            sconv.weight,
            eff_init,
            stage_slice,
            residual=sconv.residual,
            use_activation=sconv.use_activation,
            clamp_val=clamp_val,
        )

    def _get_sp_sync_slots(
        self,
        plan: _RaggedPrefillPlan,
        tpb_c: int,
        sync_all_blocks: bool,
        sp_size: int,
        device: torch.device,
    ) -> Tuple[List[Optional[torch.Tensor]], List[int]]:
        ck = (int(tpb_c), bool(sync_all_blocks), int(sp_size))
        cached = plan.sp_sync_cache.get(ck)
        if cached is not None:
            return cached
        _ = self._get_snap_schedule(plan, tpb_c, device)
        s_toks_h, s_reqs_h, _, s_ends_h, _, _, _ = plan.snap_host_cache[tpb_c]
        if not s_toks_h:
            res_empty: Tuple[List[Optional[torch.Tensor]], List[int]] = ([None] * sp_size, [0] * sp_size)
            plan.sp_sync_cache[ck] = res_empty
            return res_empty
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        bounds = [sp_slice_bounds(plan.total_tokens, sp_size, r) for r in range(sp_size)]
        per_r_slots: List[List[int]] = [[] for _ in range(sp_size)]
        for idx_s, (gtok, greq, lend) in enumerate(zip(s_toks_h, s_reqs_h, s_ends_h)):
            if not sync_all_blocks and lend != plan.active_lens[greq]:
                continue
            for r_i, (r0, r1) in enumerate(bounds):
                if r0 <= gtok < r1:
                    per_r_slots[r_i].append(idx_s)
                    break
        plan.sp_sync_host_cache[ck] = per_r_slots
        flat_slots: List[int] = []
        slot_spans: List[Optional[Tuple[int, int]]] = []
        counts: List[int] = []
        for slots in per_r_slots:
            counts.append(len(slots))
            if slots:
                a = len(flat_slots)
                flat_slots.extend(slots)
                pad = (-len(slots)) & 3
                if pad:
                    flat_slots.extend([0] * pad)
                slot_spans.append((a, a + len(slots)))
            else:
                slot_spans.append(None)
        packed_slots = _copy_i32_list_to_dev(flat_slots, device).to(torch.int64) if flat_slots else None
        per_r_dev: List[Optional[torch.Tensor]] = [
            (packed_slots[sp[0] : sp[1]] if (sp is not None and packed_slots is not None) else None)
            for sp in slot_spans
        ]
        res = (per_r_dev, counts)
        plan.sp_sync_cache[ck] = res
        return res

    def _get_or_build_prefill_sp_sync_meta(
        self,
        specs: Sequence[Tuple[Any, int, torch.Tensor, int, torch.Tensor]],
        plan: _RaggedPrefillPlan,
        sync_all_blocks: bool,
        sp_size: int,
        sp_rank: int,
        device: torch.device,
    ) -> Optional[Dict[str, Any]]:
        if not specs or not _HAS_TRITON or device.type != "cuda":
            return None
        ck = (
            int(sp_size),
            int(sp_rank),
            bool(sync_all_blocks),
            tuple(
                (p.data_ptr(), int(p.shape[0]), int(tpb_c), int(nb))
                for p, tpb_c, _, nb, _ in specs
            ),
        )
        hit = plan.sp_prefill_sync_meta_cache.get(ck)
        if hit is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            active_entries: List[Tuple[int, torch.Tensor, int, int, List[List[int]], int]] = []
            for e_raw_idx, (pool_raw, tpb_c, _, need_bytes, _) in enumerate(specs):
                self._get_sp_sync_slots(plan, tpb_c, sync_all_blocks, sp_size, device)
                s_ck = (int(tpb_c), bool(sync_all_blocks), int(sp_size))
                per_r_slots_h = plan.sp_sync_host_cache[s_ck]
                if not any(per_r_slots_h):
                    continue
                num_blocks = int(pool_raw.shape[0])
                if num_blocks <= 0 or (need_bytes % 4) != 0:
                    return None
                base_i32 = self._opaque_2d_aligned_view(pool_raw, num_blocks, need_bytes, torch.int32)
                if base_i32 is None:
                    return None
                active_entries.append((e_raw_idx, base_i32, num_blocks, need_bytes // 4, per_r_slots_h, int(base_i32.stride(0))))
            if not active_entries:
                return {"empty": True}

            block_sz = 1024
            ptrs_base_h: List[int] = []
            base_sb_h: List[int] = []
            num_blocks_h: List[int] = []
            n_elems_h: List[int] = []
            orig_indices: List[int] = []
            sample_base_i32 = active_entries[0][1]

            rank_i32 = [0] * sp_size
            g_e: List[int] = []
            g_slot: List[int] = []
            g_blk: List[int] = []
            g_off: List[int] = []
            s_e: List[int] = []
            s_slot: List[int] = []
            s_blk: List[int] = []
            s_rank: List[int] = []
            s_off: List[int] = []

            for e_i, (orig_idx, base_i32, num_blks, n_elems, per_r_slots_h, base_sb) in enumerate(active_entries):
                orig_indices.append(orig_idx)
                ptrs_base_h.append(base_i32.data_ptr())
                base_sb_h.append(base_sb)
                num_blocks_h.append(num_blks)
                n_elems_h.append(n_elems)
                n_blks = int(triton.cdiv(n_elems, block_sz))
                for r_i in range(sp_size):
                    for slot_idx in per_r_slots_h[r_i]:
                        off_i32 = rank_i32[r_i]
                        rank_i32[r_i] += n_elems
                        if r_i == sp_rank:
                            for blk in range(n_blks):
                                g_e.append(e_i)
                                g_slot.append(slot_idx)
                                g_blk.append(blk)
                                g_off.append(off_i32)
                        else:
                            for blk in range(n_blks):
                                s_e.append(e_i)
                                s_slot.append(slot_idx)
                                s_blk.append(blk)
                                s_rank.append(r_i)
                                s_off.append(off_i32)

            max_bytes = max(rank_i32) * 4 if rank_i32 else 0
            if max_bytes <= 0:
                return {"empty": True}
            max_aligned_i32 = ((max_bytes + 15) // 16) * 4

            segs_32 = [num_blocks_h, n_elems_h, g_e, g_slot, g_blk, g_off, s_e, s_slot, s_blk, s_rank, s_off]
            flat_32: List[int] = []
            spans_32: List[Tuple[int, int]] = []
            for lst in segs_32:
                s0_l = len(flat_32)
                flat_32.extend(lst)
                pad = (-len(lst)) & 3
                if pad:
                    flat_32.extend([0] * pad)
                spans_32.append((s0_l, s0_l + len(lst)))
            dev_32 = _copy_i32_list_to_dev(flat_32, device)
            v32 = [dev_32[a:b] for (a, b) in spans_32]

            ne = len(active_entries)
            arr_base_64 = array.array("q", ptrs_base_h + base_sb_h)
            dev_base_64 = _pin_stage_copy_to_device(torch.frombuffer(arr_base_64, dtype=torch.int64), device)
            ptr_or_static = (
                int(dev_base_64.data_ptr())
                | int(v32[0].data_ptr())
                | int(v32[1].data_ptr())
            )
            if g_e:
                ptr_or_static |= int(v32[2].data_ptr()) | int(v32[3].data_ptr()) | int(v32[4].data_ptr()) | int(v32[5].data_ptr())
            if s_e:
                ptr_or_static |= int(v32[6].data_ptr()) | int(v32[7].data_ptr()) | int(v32[8].data_ptr()) | int(v32[9].data_ptr()) | int(v32[10].data_ptr())

            hit = {
                "empty": False,
                "num_e": ne,
                "orig_indices": orig_indices,
                "block_sz": block_sz,
                "max_aligned_i32": max_aligned_i32,
                "sample_base": sample_base_i32,
                "ptrs_base": dev_base_64[0:ne],
                "base_sb": dev_base_64[ne : 2 * ne],
                "num_blocks": v32[0],
                "n_elems": v32[1],
                "num_gather": len(g_e),
                "g_e": v32[2],
                "g_slot": v32[3],
                "g_blk": v32[4],
                "g_off": v32[5],
                "num_scatter": len(s_e),
                "s_e": v32[6],
                "s_slot": v32[7],
                "s_blk": v32[8],
                "s_rank": v32[9],
                "s_off": v32[10],
                "ptr_or_static": ptr_or_static,
                "_refs": tuple(e[1] for e in active_entries) + (dev_32, dev_base_64),
                "last_sphys_sig": None,
                "last_sphys_dev64": None,
                "last_sphys_ptr_or": 0,
            }
            plan.sp_prefill_sync_meta_cache[ck] = hit

        if hit.get("empty"):
            return hit

        orig_indices = hit["orig_indices"]
        ne = int(hit["num_e"])
        sphys_tensors = [specs[idx][4] for idx in orig_indices]
        if any((not sp_t.is_cuda or sp_t.dtype != torch.int32 or sp_t.ndim != 1) for sp_t in sphys_tensors):
            return None
        sphys_sig = tuple((sp_t.data_ptr(), int(sp_t.stride(0))) for sp_t in sphys_tensors)
        if hit["last_sphys_sig"] != sphys_sig:
            if torch.cuda.is_current_stream_capturing():
                return None
            arr_sp_64 = array.array(
                "q",
                [p for p, _ in sphys_sig] + [s for _, s in sphys_sig],
            )
            dev_sp_64 = _pin_stage_copy_to_device(torch.frombuffer(arr_sp_64, dtype=torch.int64), device)
            hit["last_sphys_sig"] = sphys_sig
            hit["last_sphys_dev64"] = dev_sp_64
            hit["ptrs_sphys"] = dev_sp_64[0:ne]
            hit["sphys_s0"] = dev_sp_64[ne : 2 * ne]
            hit["last_sphys_ptr_or"] = int(dev_sp_64.data_ptr())
        return hit

    def _compute_snap_phys(
        self,
        plan: _RaggedPrefillPlan,
        tpb_c: int,
        bt_act: torch.Tensor,
        prefill_handle: Optional[Qwen4_5PrefillHandle],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
        max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
        s0 = int(bt_act.stride(0))
        s1 = int(bt_act.stride(1)) if bt_act.ndim >= 2 else 0
        ck = (tpb_c, bt_act.data_ptr(), max_cols, s0, s1)
        s_toks, s_reqs, s_pages, s_ends, r_sstart, s_reqs_i64, a16, a32 = self._get_snap_schedule(
            plan, tpb_c, device
        )
        if max_cols <= 0:
            vals_phys = torch.zeros_like(s_toks)
        else:
            if prefill_handle is not None and prefill_handle.is_auto_cg_dynamic_pfx and tpb_c > 0:
                s_rel = self._get_snap_rel_pages(plan, tpb_c, device)
                pfx_base_src = (
                    plan.start_pos_i64
                    if plan.bsz == 1
                    else plan.start_pos_i64.index_select(0, s_reqs_i64)
                )
                pfx_blks = torch.div(pfx_base_src, tpb_c, rounding_mode="floor")
                clamped_pages = (s_rel + pfx_blks).clamp(max=max_cols - 1)
            else:
                clamped_pages = s_pages.clamp(max=max_cols - 1)
            vals_phys = bt_act[s_reqs_i64, clamped_pages].to(torch.int32)
        if prefill_handle is not None:
            buf = prefill_handle.snap_phys_bufs.get(ck)
            if buf is None or buf.shape != vals_phys.shape or buf.device != device:
                buf = vals_phys.clone() if vals_phys.data_ptr() == 0 or max_cols > 0 else vals_phys
                prefill_handle.snap_phys_bufs[ck] = buf
            else:
                buf.copy_(vals_phys)
            s_phys = buf
        else:
            s_phys = vals_phys
        return (s_toks, s_reqs, s_ends, r_sstart, s_phys, a16, a32)

    def _compute_sp_snap_phys(
        self,
        plan: _RaggedPrefillPlan,
        sub_plan: _RaggedSPSubPlan,
        tpb_c: int,
        bt_act: torch.Tensor,
        s_phys_all: torch.Tensor,
        prefill_handle: Optional[Qwen4_5PrefillHandle],
        device: torch.device,
    ) -> torch.Tensor:
        if tpb_c not in sub_plan.snap_slot_cache:
            _ = self._get_sp_snap_base(plan, sub_plan, tpb_c, device)
        slice_ab, slot_i64 = sub_plan.snap_slot_cache[tpb_c]
        if slice_ab is not None:
            return s_phys_all[slice_ab[0] : slice_ab[1]]
        max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
        s0 = int(bt_act.stride(0))
        s1 = int(bt_act.stride(1)) if bt_act.ndim >= 2 else 0
        ck = (tpb_c, bt_act.data_ptr(), max_cols, s0, s1)
        if prefill_handle is not None:
            out_sub = prefill_handle.sp_snap_phys_bufs.get(ck)
            if out_sub is None or out_sub.shape != slot_i64.shape or out_sub.device != device:
                out_sub = s_phys_all.index_select(0, slot_i64)
                prefill_handle.sp_snap_phys_bufs[ck] = out_sub
            else:
                torch.index_select(s_phys_all, 0, slot_i64, out=out_sub)
            return out_sub
        return s_phys_all.index_select(0, slot_i64)

    def _prewarm_prefill_sp_sync_meta(
        self,
        model: Qwen4_5Model,
        kv_cache: Any,
        attn_inputs: Any,
        plan: _RaggedPrefillPlan,
        sub_plan: _RaggedSPSubPlan,
        prefill_handle: Optional[Qwen4_5PrefillHandle],
        *,
        sync_all_blocks: bool,
        sp_size: int,
        sp_rank: int,
        device: torch.device,
    ) -> None:
        if kv_cache is None or sp_size <= 1 or plan.total_tokens <= 0:
            return
        dtype_bytes = _dtype_itemsize(model.embed_tokens.weight.dtype)
        specs: List[Tuple[Any, int, torch.Tensor, int, torch.Tensor]] = []
        bt_cache: Dict[str, torch.Tensor] = {}

        def _bt_for(tag_name: str) -> torch.Tensor:
            bt_v = bt_cache.get(tag_name)
            if bt_v is None:
                bt_v = _block_table_for_tag(attn_inputs, tag_name)
                if plan.active_reqs and len(plan.active_reqs) < plan.bsz:
                    assert plan.act_idx_dev is not None
                    bt_v = bt_v.index_select(0, plan.act_idx_dev)
                bt_cache[tag_name] = bt_v
            return bt_v

        snap_cache_local: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}

        def _sp_phys(cache: Any, bt_act: torch.Tensor) -> torch.Tensor:
            tpb_c = int(cache.seq_size_per_block)
            max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
            bs0 = int(bt_act.stride(0))
            bs1 = int(bt_act.stride(1)) if bt_act.ndim >= 2 else 0
            ck = (tpb_c, bt_act.data_ptr(), max_cols, bs0, bs1)
            hit_s = snap_cache_local.get(ck)
            if hit_s is not None:
                return hit_s
            _, _, _, _, s_phys_all, _, _ = self._compute_snap_phys(plan, tpb_c, bt_act, prefill_handle, device)
            snap_cache_local[ck] = s_phys_all
            return s_phys_all

        for layer in model.layers:
            lid = layer.layer_idx
            lc = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}
            if layer.ple is not None and PLE_STATE_TAG in lc and PLE_NGRAM_CTX_TAG in lc:
                c_st, c_ctx = lc[PLE_STATE_TAG], lc[PLE_NGRAM_CTX_TAG]
                bt_st, bt_ctx = _bt_for(PLE_STATE_TAG), _bt_for(PLE_NGRAM_CTX_TAG)
                specs.append((c_st.kv_cache_base, int(c_st.seq_size_per_block), bt_st, self.cfg.hc_count * self.cfg.hidden_size * max(0, self.cfg.ple_conv_kernel_size - 1) * dtype_bytes, _sp_phys(c_st, bt_st)))
                specs.append((c_ctx.kv_cache_base, int(c_ctx.seq_size_per_block), bt_ctx, max(0, self.cfg.ngram_size - 1) * 8, _sp_phys(c_ctx, bt_ctx)))
            if layer.attn_output_sconv is not None:
                tag_out = LINEAR_ATTN_OUTPUT_SCONV_TAG if layer.layer_type == "linear_attention" else FULL_ATTN_OUTPUT_SCONV_TAG
                if tag_out in lc:
                    c_ao = lc[tag_out]
                    bt_ao = _bt_for(tag_out)
                    specs.append((c_ao.kv_cache_base, int(c_ao.seq_size_per_block), bt_ao, self.cfg.hidden_size * max(0, layer.attn_output_sconv.kernel_size - 1) * dtype_bytes, _sp_phys(c_ao, bt_ao)))
            if layer.ffn_output_sconv is not None and FFN_OUTPUT_SCONV_TAG in lc:
                c_fo = lc[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _bt_for(FFN_OUTPUT_SCONV_TAG)
                specs.append((c_fo.kv_cache_base, int(c_fo.seq_size_per_block), bt_fo, self.cfg.hidden_size * max(0, layer.ffn_output_sconv.kernel_size - 1) * dtype_bytes, _sp_phys(c_fo, bt_fo)))
        if not specs:
            return
        eff_sync_all = bool(sync_all_blocks)
        if not eff_sync_all:
            total_est = 0
            for _, tpb_c, _, nb, _ in specs:
                _, _, _, s_ends_h, _, _, _ = plan.snap_host_cache[tpb_c]
                total_est += plan.bsz * nb
            if total_est <= 8 * 1024 * 1024:
                eff_sync_all = True
        _ = self._get_or_build_prefill_sp_sync_meta(specs, plan, eff_sync_all, sp_size, sp_rank, device)

    def _sync_sp_prefill_snapshots(
        self,
        specs: Sequence[Tuple[Any, int, torch.Tensor, int, torch.Tensor]],
        plan: _RaggedPrefillPlan,
        *,
        sync_all_blocks: bool,
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any,
        device: torch.device,
    ) -> None:
        """Synchronize SP-written PLE & output ShortConv snapshots across all SP ranks (CG-safe)."""
        if not specs or sp_size <= 1:
            return
        total_est = 0
        for pool_raw, tpb_c, bt_act, need_bytes, s_phys in specs:
            _, _, _, s_ends_h, _, _, _ = plan.snap_host_cache[tpb_c]
            cnt = len(s_ends_h) if sync_all_blocks else plan.bsz
            total_est += cnt * need_bytes
        if not sync_all_blocks and total_est <= 8 * 1024 * 1024:
            sync_all_blocks = True

        meta = self._get_or_build_prefill_sp_sync_meta(
            specs, plan, sync_all_blocks, sp_size, sp_rank, device
        )
        if meta is not None:
            if meta.get("empty"):
                return
            max_aligned_i32 = int(meta["max_aligned_i32"])
            num_gather = int(meta["num_gather"])
            num_scatter = int(meta["num_scatter"])
            send_buf = torch.empty((max_aligned_i32,), dtype=torch.int32, device=device)
            ptr_or_base = int(meta["ptr_or_static"]) | int(meta["last_sphys_ptr_or"])
            if num_gather > 0:
                ckey_g = (
                    device.index,
                    "pre_sp_gath",
                    meta["num_e"],
                    _triton_int_spec(num_gather),
                    meta["block_sz"],
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE,
                    ckey_g,
                    _prefill_sp_gather_opaque_batched_triton,
                    num_gather,
                    1,
                    1,
                    ptr_or_base | send_buf.data_ptr(),
                    (
                        meta["g_e"],
                        meta["g_slot"],
                        meta["g_blk"],
                        meta["g_off"],
                        meta["ptrs_base"],
                        meta["ptrs_sphys"],
                        meta["base_sb"],
                        meta["sphys_s0"],
                        meta["num_blocks"],
                        meta["n_elems"],
                        send_buf,
                    ),
                    {"BLOCK_SIZE": meta["block_sz"]},
                    num_warps=4,
                )
            all_bf16 = self._sp_all_gather_dim0(
                send_buf.view(torch.bfloat16).unsqueeze(0),
                total_tokens=sp_size,
                sp_size=sp_size,
                sp_rank=sp_rank,
                sp_group=sp_group,
                sp_comm=sp_comm,
            )
            if num_scatter > 0:
                all_buf = all_bf16.contiguous().view(torch.int32).reshape(sp_size, max_aligned_i32)
                all_s0 = int(all_buf.stride(0))
                ckey_s = (
                    device.index,
                    "pre_sp_scat",
                    meta["num_e"],
                    _triton_int_spec(num_scatter),
                    _triton_int_spec(all_s0),
                    meta["block_sz"],
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE,
                    ckey_s,
                    _prefill_sp_scatter_opaque_batched_triton,
                    num_scatter,
                    1,
                    1,
                    ptr_or_base | all_buf.data_ptr(),
                    (
                        meta["s_e"],
                        meta["s_slot"],
                        meta["s_blk"],
                        meta["s_rank"],
                        meta["s_off"],
                        meta["ptrs_base"],
                        meta["ptrs_sphys"],
                        meta["base_sb"],
                        meta["sphys_s0"],
                        meta["num_blocks"],
                        meta["n_elems"],
                        all_buf,
                        all_s0,
                    ),
                    {"BLOCK_SIZE": meta["block_sz"]},
                    num_warps=4,
                )
            return

        by_rank_slices: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]] = [[] for _ in range(sp_size)]
        rank_bytes = [0] * sp_size
        for pool_raw, tpb_c, bt_act, need_bytes, s_phys in specs:
            per_r_dev, per_r_counts = self._get_sp_sync_slots(plan, tpb_c, sync_all_blocks, sp_size, device)
            if not any(per_r_counts):
                continue
            pool_u8 = pool_raw.reshape(int(pool_raw.shape[0]), -1).view(torch.uint8)
            for r_i in range(sp_size):
                idx_t = per_r_dev[r_i]
                cnt = per_r_counts[r_i]
                if idx_t is not None and cnt > 0:
                    rows_p = s_phys.index_select(0, idx_t)
                    by_rank_slices[r_i].append((pool_raw, pool_u8, rows_p, need_bytes))
                    rank_bytes[r_i] += cnt * need_bytes

        max_bytes = max(rank_bytes) if rank_bytes else 0
        if max_bytes <= 0:
            return
        max_aligned = ((max_bytes + 15) // 16) * 16
        local_chunks: List[torch.Tensor] = []
        for pool_raw, pool_u8, rows_p, nb in by_rank_slices[sp_rank]:
            valid_p = rows_p.to(torch.int64).clamp(min=0, max=int(pool_u8.shape[0]) - 1)
            local_chunks.append(pool_u8.index_select(0, valid_p)[:, :nb].reshape(-1))
        send_buf = torch.zeros((max_aligned,), dtype=torch.uint8, device=device)
        if local_chunks:
            packed_loc = local_chunks[0] if len(local_chunks) == 1 else torch.cat(local_chunks, dim=0)
            send_buf[: int(packed_loc.numel())].copy_(packed_loc)

        all_buf = self._sp_all_gather_dim0(
            send_buf,
            total_tokens=sp_size * max_aligned,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_comm=sp_comm,
        ).view(sp_size, max_aligned)

        for r_i in range(sp_size):
            if r_i == sp_rank or rank_bytes[r_i] == 0:
                continue
            off = 0
            r_flat = all_buf[r_i]
            for pool_raw, pool_u8, rows_p, nb in by_rank_slices[r_i]:
                n_r = int(rows_p.numel())
                seg = r_flat[off : off + n_r * nb].view(n_r, nb)
                off += n_r * nb
                if device.type == "cuda" and _HAS_TRITON:
                    _write_opaque_rows_raw_gpu(
                        pool_raw,
                        rows_p,
                        seg,
                        tag="sp_prefill_snap",
                        layer_idx=-1,
                    )
                else:
                    rows_i64 = rows_p.to(torch.int64)
                    mask = (rows_i64 > 0) & (rows_i64 < int(pool_u8.shape[0]))
                    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if int(valid_idx.numel()) > 0:
                        dst_rows = rows_i64.index_select(0, valid_idx)
                        pool_u8[:, :nb].index_copy_(0, dst_rows, seg.index_select(0, valid_idx))

    def _write_opaque_batch_gpu(
        self,
        cache: Any,
        block_table: torch.Tensor,
        seq_before_dev: torch.Tensor,
        tensor: torch.Tensor,
        tag: str,
        layer_idx: int,
    ) -> None:
        if not _HAS_TRITON:
            raise RuntimeError("Triton is required for GPU opaque state writes")
        base = cache.kv_cache_base
        tpb = int(cache.seq_size_per_block)
        bsz = int(seq_before_dev.shape[0])
        num_blocks = int(base.shape[0])
        max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
        if bsz == 0 or max_blocks == 0 or num_blocks == 0:
            return

        src_c = tensor.contiguous().reshape(bsz, -1)
        src_u8 = src_c.view(torch.uint8)
        need_bytes = int(src_u8.shape[1])
        base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
        if int(base_u8.shape[1]) < need_bytes:
            raise RuntimeError(
                f"Opaque pool row too small for write: {base_u8.shape[1]} bytes < {need_bytes} bytes "
                f"for tag={tag} layer={layer_idx}"
            )

        dst_view = self._opaque_2d_aligned_view(base, num_blocks, need_bytes, torch.int64)
        src_view = self._opaque_2d_aligned_view(src_c, bsz, need_bytes, torch.int64) if dst_view is not None else None
        if dst_view is not None and src_view is not None:
            n_elems = need_bytes // 8
        else:
            dst_view = self._opaque_2d_aligned_view(base, num_blocks, need_bytes, torch.int32)
            src_view = self._opaque_2d_aligned_view(src_c, bsz, need_bytes, torch.int32) if dst_view is not None else None
            if dst_view is not None and src_view is not None:
                n_elems = need_bytes // 4
            else:
                dst_view = base_u8
                src_view = src_u8
                n_elems = need_bytes

        block_sz = 1024
        dst_s0, src_s0 = int(dst_view.stride(0)), int(src_view.stride(0))
        bt_s0, bt_s1 = int(block_table.stride(0)), int(block_table.stride(1))
        ptr_or = (
            dst_view.data_ptr() | src_view.data_ptr()
            | block_table.data_ptr() | seq_before_dev.data_ptr()
        )
        ckey = (
            dst_view.device.index, "rt_write_op_paged", dst_view.dtype, src_view.dtype,
            block_table.dtype, seq_before_dev.dtype,
            _triton_int_spec(dst_s0), _triton_int_spec(src_s0),
            _triton_int_spec(bt_s0), _triton_int_spec(bt_s1),
            _triton_int_spec(tpb), _triton_int_spec(max_blocks),
            _triton_int_spec(num_blocks), _triton_int_spec(n_elems),
            block_sz,
        )
        triton_fast_launch(
            _RUNTIME_TRITON_CACHE, ckey, _write_opaque_rows_paged_triton,
            bsz, int(triton.cdiv(n_elems, block_sz)), 1, ptr_or,
            (dst_view, src_view, block_table, seq_before_dev,
             dst_s0, src_s0, bt_s0, bt_s1, tpb, max_blocks, num_blocks, n_elems),
            {"BLOCK_SIZE": block_sz},
            num_warps=4,
        )

    def _collect_dec_sp_specs(
        self,
        model: Qwen4_5Model,
        kv_cache: Any,
        attention_inputs: Any,
        dtype: torch.dtype,
    ) -> List[Tuple[Any, torch.Tensor, int]]:
        specs: List[Tuple[Any, torch.Tensor, int]] = []
        d_bytes = _dtype_itemsize(dtype)
        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}
            if layer.ple is not None:
                c_st = layer_caches[PLE_STATE_TAG]
                c_ctx = layer_caches[PLE_NGRAM_CTX_TAG]
                bt_st = _block_table_for_tag(attention_inputs, PLE_STATE_TAG)
                bt_ctx = _block_table_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
                st_b = math.prod(self._opaque_shapes[PLE_STATE_TAG]) * d_bytes
                ctx_b = math.prod(self._opaque_shapes[PLE_NGRAM_CTX_TAG]) * 8
                if st_b > 0:
                    specs.append((c_st, bt_st, st_b))
                if ctx_b > 0:
                    specs.append((c_ctx, bt_ctx, ctx_b))
            if layer.attn_output_sconv is not None:
                tag_ao = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao = layer_caches[tag_ao]
                bt_ao = _block_table_for_tag(attention_inputs, tag_ao)
                ao_b = self.cfg.hidden_size * max(0, int(layer.attn_output_sconv.kernel_size) - 1) * d_bytes
                if ao_b > 0:
                    specs.append((c_ao, bt_ao, ao_b))
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                fo_b = self.cfg.hidden_size * max(0, int(layer.ffn_output_sconv.kernel_size) - 1) * d_bytes
                if fo_b > 0:
                    specs.append((c_fo, bt_fo, fo_b))
        return specs

    def _get_or_build_dec_sp_meta(
        self,
        specs: Sequence[Tuple[Any, torch.Tensor, int]],
        bsz: int,
        dev: torch.device,
    ) -> Optional[Dict[str, Any]]:
        if not specs or not _HAS_TRITON or dev.type != "cuda":
            return None
        dev_idx = dev.index if dev.index is not None else -1
        key = (
            dev_idx,
            int(bsz),
            tuple(
                (
                    c.kv_cache_base.data_ptr(),
                    int(c.kv_cache_base.shape[0]),
                    max(1, int(c.seq_size_per_block)),
                    bt.data_ptr(),
                    int(bt.stride(0)),
                    int(bt.stride(1)) if bt.ndim >= 2 else 0,
                    int(bt.shape[1]) if bt.ndim >= 2 else 0,
                    int(nb),
                )
                for c, bt, nb in specs
            ),
        )
        hit = self._dec_sp_meta_cache.get(key)
        if hit is not None:
            self._dec_sp_meta_cache.move_to_end(key)
            return hit
        if torch.cuda.is_current_stream_capturing():
            return None

        ptrs_base_h: List[int] = []
        ptrs_bt_h: List[int] = []
        base_sb_h: List[int] = []
        bt_s0_h: List[int] = []
        bt_s1_h: List[int] = []
        tpb_h: List[int] = []
        max_blocks_h: List[int] = []
        num_blocks_h: List[int] = []
        n_elems_h: List[int] = []
        col_off_h: List[int] = []
        tile_e_h: List[int] = []
        tile_blk_h: List[int] = []
        sample_base_i32: Optional[torch.Tensor] = None
        block_sz = 1024
        col_cur = 0

        for e_idx, (c, bt, nb) in enumerate(specs):
            base = c.kv_cache_base
            num_blocks = int(base.shape[0])
            max_blocks = int(bt.shape[1]) if bt.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0 or bt.dtype != torch.int32 or (nb % 4) != 0:
                return None
            base_i32 = self._opaque_2d_aligned_view(base, num_blocks, nb, torch.int32)
            if base_i32 is None:
                return None
            if sample_base_i32 is None:
                sample_base_i32 = base_i32
            n_elems = nb // 4
            n_blks = int(triton.cdiv(n_elems, block_sz))
            ptrs_base_h.append(base_i32.data_ptr())
            ptrs_bt_h.append(bt.data_ptr())
            base_sb_h.append(int(base_i32.stride(0)))
            bt_s0_h.append(int(bt.stride(0)))
            bt_s1_h.append(int(bt.stride(1)) if bt.ndim >= 2 else 0)
            tpb_h.append(max(1, int(c.seq_size_per_block)))
            max_blocks_h.append(max_blocks)
            num_blocks_h.append(num_blocks)
            n_elems_h.append(n_elems)
            col_off_h.append(col_cur)
            for b_i in range(n_blks):
                tile_e_h.append(e_idx)
                tile_blk_h.append(b_i)
            col_cur += n_elems

        if not tile_e_h or col_cur <= 0:
            return None
        row_stride = ((col_cur + 3) // 4) * 4

        flat_32: List[int] = []
        spans_32: List[Tuple[int, int]] = []
        for lst in (
            tile_e_h,
            tile_blk_h,
            tpb_h,
            max_blocks_h,
            num_blocks_h,
            n_elems_h,
            col_off_h,
        ):
            s0 = len(flat_32)
            flat_32.extend(lst)
            pad = (-len(lst)) & 3
            if pad:
                flat_32.extend([0] * pad)
            spans_32.append((s0, s0 + len(lst)))
        dev_32 = _copy_i32_list_to_dev(flat_32, dev)
        v32 = [dev_32[a:b] for (a, b) in spans_32]

        flat_64 = ptrs_base_h + ptrs_bt_h + base_sb_h + bt_s0_h + bt_s1_h
        arr_64 = array.array("q", flat_64)
        dev_64 = _pin_stage_copy_to_device(torch.frombuffer(arr_64, dtype=torch.int64), dev)
        ne = len(specs)
        meta = {
            "sample_base": sample_base_i32,
            "num_e": ne,
            "num_tiles": len(tile_e_h),
            "row_stride": row_stride,
            "block_sz": block_sz,
            "tile_e": v32[0],
            "tile_blk": v32[1],
            "tpb": v32[2],
            "max_blocks": v32[3],
            "num_blocks": v32[4],
            "n_elems": v32[5],
            "col_off": v32[6],
            "ptrs_base": dev_64[0:ne],
            "ptrs_bt": dev_64[ne : 2 * ne],
            "base_sb": dev_64[2 * ne : 3 * ne],
            "bt_s0": dev_64[3 * ne : 4 * ne],
            "bt_s1": dev_64[4 * ne : 5 * ne],
            "ptr_or_meta": (
                v32[0].data_ptr()
                | v32[1].data_ptr()
                | dev_64[0:ne].data_ptr()
                | dev_64[ne : 2 * ne].data_ptr()
                | dev_64[2 * ne : 3 * ne].data_ptr()
                | dev_64[3 * ne : 4 * ne].data_ptr()
                | dev_64[4 * ne : 5 * ne].data_ptr()
                | v32[2].data_ptr()
                | v32[3].data_ptr()
                | v32[4].data_ptr()
                | v32[5].data_ptr()
                | v32[6].data_ptr()
            ),
            "_refs": tuple((c.kv_cache_base, bt) for c, bt, _ in specs),
        }
        self._dec_sp_meta_cache[key] = meta
        if len(self._dec_sp_meta_cache) > 64:
            self._dec_sp_meta_cache.popitem(last=False)
        return meta

    def _prewarm_dec_sp_meta(
        self,
        model: Qwen4_5Model,
        kv_cache: Any,
        attention_inputs: Any,
        bsz: int,
        dev: torch.device,
    ) -> None:
        if dev.type != "cuda" or bsz <= 0:
            return
        specs = self._collect_dec_sp_specs(
            model, kv_cache, attention_inputs, model.embed_tokens.weight.dtype
        )
        _ = self._get_or_build_dec_sp_meta(specs, bsz, dev)

    def _sync_sp_decode_opaque_states(
        self,
        specs: Sequence[Tuple[Any, torch.Tensor, int]],
        seq_lens_before_dev: torch.Tensor,
        *,
        bsz: int,
        b0: int,
        b1: int,
        sp_size: int,
        sp_rank: int,
        sp_group: Any,
        sp_comm: Any,
        device: torch.device,
    ) -> None:
        """Synchronize Batch-SP opaque decode states across SP ranks (1 AllGather, CG-safe)."""
        if not specs or sp_size <= 1 or bsz <= 0 or b1 <= b0:
            return
        b_loc = b1 - b0
        meta = self._get_or_build_dec_sp_meta(specs, bsz, device)
        if meta is not None:
            row_stride = int(meta["row_stride"])
            send_buf = torch.zeros((b_loc, row_stride), dtype=torch.int32, device=device)
            send_s0 = int(send_buf.stride(0))
            ptr_or_base = int(meta["ptr_or_meta"]) | seq_lens_before_dev.data_ptr()
            ptr_or_g = ptr_or_base | send_buf.data_ptr()
            ckey_g = (
                device.index,
                "dec_sp_gath",
                meta["num_e"],
                _triton_int_spec(meta["num_tiles"]),
                _triton_int_spec(send_s0),
                meta["block_sz"],
            )
            triton_fast_launch(
                _RUNTIME_TRITON_CACHE,
                ckey_g,
                _dec_sp_gather_opaque_batched_triton,
                meta["num_tiles"],
                b_loc,
                1,
                ptr_or_g,
                (
                    meta["tile_e"],
                    meta["tile_blk"],
                    meta["ptrs_base"],
                    meta["ptrs_bt"],
                    meta["base_sb"],
                    meta["bt_s0"],
                    meta["bt_s1"],
                    meta["tpb"],
                    meta["max_blocks"],
                    meta["num_blocks"],
                    meta["n_elems"],
                    meta["col_off"],
                    seq_lens_before_dev,
                    send_buf,
                    send_s0,
                    b0,
                ),
                {"BLOCK_SIZE": meta["block_sz"]},
                num_warps=4,
            )

            all_bf16 = self._sp_all_gather_dim0(
                send_buf.view(torch.bfloat16),
                total_tokens=bsz,
                sp_size=sp_size,
                sp_rank=sp_rank,
                sp_group=sp_group,
                sp_comm=sp_comm,
            )
            all_buf = all_bf16.contiguous().view(torch.int32).reshape(bsz, row_stride)
            all_s0 = int(all_buf.stride(0))
            ptr_or_s = ptr_or_base | all_buf.data_ptr()
            ckey_s = (
                device.index,
                "dec_sp_scat",
                meta["num_e"],
                _triton_int_spec(meta["num_tiles"]),
                _triton_int_spec(all_s0),
                meta["block_sz"],
            )
            triton_fast_launch(
                _RUNTIME_TRITON_CACHE,
                ckey_s,
                _dec_sp_scatter_opaque_batched_triton,
                meta["num_tiles"],
                bsz,
                1,
                ptr_or_s,
                (
                    meta["tile_e"],
                    meta["tile_blk"],
                    meta["ptrs_base"],
                    meta["ptrs_bt"],
                    meta["base_sb"],
                    meta["bt_s0"],
                    meta["bt_s1"],
                    meta["tpb"],
                    meta["max_blocks"],
                    meta["num_blocks"],
                    meta["n_elems"],
                    meta["col_off"],
                    seq_lens_before_dev,
                    all_buf,
                    all_s0,
                    b0,
                    b1,
                ),
                {"BLOCK_SIZE": meta["block_sz"]},
                num_warps=4,
            )
            return

        # Portable fallback (CPU or unaligned opaque row sizes)
        row_b = sum(nb for _, _, nb in specs)
        row_aligned = ((row_b + 15) // 16) * 16
        send_u8 = torch.zeros((b_loc, row_aligned), dtype=torch.uint8, device=device)
        off = 0
        for c, bt, nb in specs:
            base = c.kv_cache_base
            num_blocks = int(base.shape[0])
            tpb = max(1, int(c.seq_size_per_block))
            max_blocks = int(bt.shape[1]) if bt.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0:
                off += nb
                continue
            base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
            pos_loc = seq_lens_before_dev[b0:b1].to(torch.int64)
            pages = (pos_loc.clamp(min=0) // tpb).clamp(max=max_blocks - 1)
            phys = bt[b0:b1].gather(1, pages.unsqueeze(1)).squeeze(1).to(torch.int64)
            valid = (pos_loc >= 0) & (phys > 0) & (phys < num_blocks)
            safe_p = torch.where(valid, phys, 0)
            rows_u8 = base_u8.index_select(0, safe_p)[:, :nb]
            send_u8[:, off : off + nb].copy_(torch.where(valid.unsqueeze(1), rows_u8, 0))
            off += nb

        all_u8 = self._sp_all_gather_dim0(
            send_u8,
            total_tokens=bsz,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_comm=sp_comm,
        ).reshape(bsz, row_aligned)
        off = 0
        for c, bt, nb in specs:
            base = c.kv_cache_base
            num_blocks = int(base.shape[0])
            tpb = max(1, int(c.seq_size_per_block))
            max_blocks = int(bt.shape[1]) if bt.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0:
                off += nb
                continue
            pos_all = seq_lens_before_dev.to(torch.int64)
            pages = (pos_all.clamp(min=0) // tpb).clamp(max=max_blocks - 1)
            phys = bt.gather(1, pages.unsqueeze(1)).squeeze(1).to(torch.int64)
            valid = (pos_all >= 0) & (phys > 0) & (phys < num_blocks)
            phys_rem = phys.clone()
            phys_rem[b0:b1] = 0
            phys_rem = torch.where(valid, phys_rem, 0)
            seg = all_u8[:, off : off + nb].contiguous()
            off += nb
            if device.type == "cuda" and _HAS_TRITON:
                _write_opaque_rows_raw_gpu(base, phys_rem.to(torch.int32), seg, tag="dec_sp_sync", layer_idx=-1)
            else:
                base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
                v_idx = torch.nonzero(phys_rem > 0, as_tuple=False).squeeze(-1)
                if int(v_idx.numel()) > 0:
                    base_u8[:, :nb].index_copy_(0, phys_rem.index_select(0, v_idx), seg.index_select(0, v_idx))

    def _run_decode_batch_gpu(
        self,
        model: Qwen4_5Model,
        input_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
        *,
        seq_lens_before_dev: torch.Tensor,
        decode_attn_handle: Any,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """CUDA-graph-safe zero-sync Decode path backed by FlashInfer/QSA & Triton."""
        device = input_ids_2d.device
        dtype = model.embed_tokens.weight.dtype
        bsz = int(input_ids_2d.shape[0])
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )
        if bsz == 0:
            self._run_empty_ep_collectives(model, dtype, device)
            empty_h = torch.empty((0, self.cfg.hidden_size), dtype=dtype, device=device)
            if capture_layer_ids:
                empty_aux = torch.empty((0, len(capture_layer_ids) * self.cfg.hidden_size), dtype=dtype, device=device)
                return empty_h, empty_aux
            return empty_h

        eff_sp, eff_rk, eff_grp, eff_comm = self._resolve_sp_dist()
        dec_sp_active = eff_sp > 1 and bsz >= eff_sp and (bsz % eff_sp) == 0
        if dec_sp_active:
            from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

            b0, b1 = sp_slice_bounds(bsz, eff_sp, eff_rk)
            b_loc = b1 - b0
            ids_loc = input_ids_2d[b0:b1]
            seq_loc_dev = seq_lens_before_dev[b0:b1]
        else:
            b0, b1, b_loc = 0, bsz, bsz
            ids_loc = input_ids_2d
            seq_loc_dev = seq_lens_before_dev

        emb = model.embed_tokens(ids_loc)
        hyper_hidden = (
            emb
            if self.cfg.hc_count > 1 and (len(model.layers) == 0 or model.layers[0].ple is None)
            else emb.repeat(1, 1, self.cfg.hc_count)
        )
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        deferred_sconv: Optional[Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[float]]] = None
        shared_gdn: Dict[int, Any] = {}
        full_kv_layer_caches: Dict[int, Any] = {}
        source_idx_caches: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        qsa_vis_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        cp_idx_kw = self._cp_indexer_kwargs()
        seq_plus1_dev: Optional[torch.Tensor] = None

        def _get_qsa_vis_dec(bt_chk: torch.Tensor, use_pad: bool) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
            nonlocal seq_plus1_dev
            if not use_pad:
                if seq_plus1_dev is None:
                    seq_plus1_dev = seq_lens_before_dev + 1
                return None, seq_plus1_dev
            key = bt_chk.data_ptr()
            cached = qsa_vis_cache.get(key)
            if cached is not None:
                return cached
            if seq_plus1_dev is None:
                seq_plus1_dev = seq_lens_before_dev + 1
            act = (bt_chk != 0).any(dim=1)
            vis = torch.where(act, seq_plus1_dev, 0)
            qsa_vis_cache[key] = (act, vis)
            return act, vis

        def _flush_deferred_sconv() -> None:
            nonlocal deferred, deferred_sconv
            if deferred_sconv is None:
                return
            s_out, s_inj, c_sc, bt_sc, mod_sc, c_val = deferred_sconv
            sc_pool_bf16, _ = _view_opaque_pool_2d(c_sc.kv_cache_base, dtype)
            conv_y = causal_dwconv1d_decode_opaque_triton(
                s_out,
                mod_sc.weight,
                sc_pool_bf16,
                bt_sc,
                seq_loc_dev,
                page_size=int(c_sc.seq_size_per_block),
                pool_col_offset=0,
                residual=mod_sc.residual,
                use_activation=mod_sc.use_activation,
                clamp_val=c_val,
            )
            deferred = (conv_y, s_inj)
            deferred_sconv = None
        dec_sp_sync_specs: List[Tuple[Any, torch.Tensor, int]] = []
        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None
        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = torch.empty(
                (b_loc, len(capture_layer_ids) * self.cfg.hidden_size),
                dtype=dtype,
                device=device,
            )
        decode_qsa_max_ctx_len: Optional[int] = None
        if (
            self.cfg.has_qsa
            and isinstance(decode_attn_handle, Qwen4_5QSADecodeHandle)
            and not decode_attn_handle.is_cuda_graph
        ):
            host_seq = getattr(decode_attn_handle.full_inputs, "sequence_lengths", None)
            if (
                isinstance(host_seq, torch.Tensor)
                and host_seq.device.type == "cpu"
                and host_seq.numel() > 0
            ):
                host_max_vis = int(host_seq.max().item()) + 1
                decode_qsa_max_ctx_len = max(
                    1, host_max_vis // self.cfg.indexer_compress_ratio
                )

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}

            # 1. Optional PLE
            if layer.ple is not None:
                _flush_deferred_sconv()
                if deferred is not None:
                    hyper_hidden = layer.attn_hyper_connection.combine(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                    deferred = None
                ple_st_cache = layer_caches[PLE_STATE_TAG]
                ple_ctx_cache = layer_caches[PLE_NGRAM_CTX_TAG]
                bt_st = _block_table_for_tag(attention_inputs, PLE_STATE_TAG)
                bt_ctx = _block_table_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
                bt_st_loc = bt_st[b0:b1] if dec_sp_active else bt_st
                bt_ctx_loc = bt_ctx[b0:b1] if dec_sp_active else bt_ctx
                ple_state = PLELayerState(
                    ngram_context=self._read_opaque_batch_gpu(
                        ple_ctx_cache,
                        bt_ctx_loc,
                        seq_loc_dev,
                        PLE_NGRAM_CTX_TAG,
                        lid,
                        fill_value=self.cfg.eos_token_id,
                    ),
                    conv_state=self._read_opaque_batch_gpu(
                        ple_st_cache,
                        bt_st_loc,
                        seq_loc_dev,
                        PLE_STATE_TAG,
                        lid,
                        target_dtype=dtype,
                    ),
                )
                hyper_hidden = layer.ple(
                    hyper_hidden,
                    ids_loc,
                    state=ple_state,
                    sp_tokens_sharded=dec_sp_active,
                    sp_total_tokens=bsz if dec_sp_active else None,
                    sp_size=eff_sp if dec_sp_active else None,
                    sp_rank=eff_rk if dec_sp_active else None,
                    sp_group=eff_grp if dec_sp_active else None,
                    sp_fast_comm=eff_comm if dec_sp_active else None,
                )
                self._write_opaque_batch_gpu(
                    ple_st_cache, bt_st_loc, seq_loc_dev, ple_state.conv_state, PLE_STATE_TAG, lid
                )
                self._write_opaque_batch_gpu(
                    ple_ctx_cache,
                    bt_ctx_loc,
                    seq_loc_dev,
                    ple_state.ngram_context,
                    PLE_NGRAM_CTX_TAG,
                    lid,
                )
                if dec_sp_active:
                    d_bytes = _dtype_itemsize(dtype)
                    st_b = math.prod(self._opaque_shapes[PLE_STATE_TAG]) * d_bytes
                    ctx_b = math.prod(self._opaque_shapes[PLE_NGRAM_CTX_TAG]) * 8
                    if st_b > 0:
                        dec_sp_sync_specs.append((ple_st_cache, bt_st, st_b))
                    if ctx_b > 0:
                        dec_sp_sync_specs.append((ple_ctx_cache, bt_ctx, ctx_b))

            # 2. Attn Hyper-Connection (fuse deferred FFN ShortConv + combine + next HC norm)
            if deferred_sconv is not None:
                s_out, s_inj, c_sc, bt_sc, mod_sc, c_val = deferred_sconv
                attn_hc = layer.attn_hyper_connection
                if (
                    s_out.is_cuda
                    and dtype == torch.bfloat16
                    and mod_sc.kernel_size == 4
                    and attn_hc.use_combine
                ):
                    sc_pool_bf16, _ = _view_opaque_pool_2d(c_sc.kv_cache_base, dtype)
                    hyper_hidden, attn_normed = hc_sconv_decode_opaque_combine_norm_triton(
                        s_out,
                        hyper_hidden,
                        s_inj,
                        attn_hc.hc_norm_weight,
                        attn_hc.hc_norm_bias,
                        mod_sc.weight,
                        sc_pool_bf16,
                        bt_sc,
                        seq_loc_dev,
                        page_size=int(c_sc.seq_size_per_block),
                        eps=attn_hc.eps,
                        conv_residual=mod_sc.residual,
                        conv_use_silu=mod_sc.use_activation,
                        clamp_val=c_val,
                    )
                    attn_in, hyper_hidden, attn_inj, prev_latent = attn_hc.mix_from_normed(
                        attn_normed, hyper_hidden, prev_latent
                    )
                    deferred_sconv = None
                else:
                    _flush_deferred_sconv()
                    assert deferred is not None
                    attn_in, hyper_hidden, attn_inj, prev_latent = (
                        attn_hc.combine_and_mix_with_latent(
                            deferred[0], hyper_hidden, deferred[1], prev_latent
                        )
                    )
                    deferred = None
            elif deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.mix_with_latent(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.combine_and_mix_with_latent(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
                deferred = None
            if aux_out is not None and lid in cap_slots:
                attn_flat = attn_in.reshape(b_loc, self.cfg.hidden_size)
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * self.cfg.hidden_size : (slot_i + 1) * self.cfg.hidden_size].copy_(
                        attn_flat
                    )

            if dec_sp_active:
                attn_in = self._sp_all_gather_dim0(
                    attn_in.reshape(b_loc, self.cfg.hidden_size),
                    total_tokens=bsz,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                ).view(bsz, 1, self.cfg.hidden_size)

            # 3. Attention (Linear GDN or Full MQA-1)
            if layer.layer_type == "linear_attention":
                assert layer.linear_attn is not None
                la = layer.linear_attn
                can_gdn_opaque = (
                    _HAS_TRITON
                    and attn_in.is_cuda
                    and dtype == torch.bfloat16
                    and la.conv_kernel_size == 4
                    and la.head_k_dim == 64
                    and la.head_v_dim == 64
                )
                if (
                    can_gdn_opaque
                    and (
                        LINEAR_ATTN_TAG in layer_caches
                        or (
                            GDN_SSM_STATE_TAG in layer_caches
                            and (
                                GDN_CONV_STATE_TAG in layer_caches
                                or (layer.is_consumer and GDN_CONSUMER_Q_CONV_TAG in layer_caches)
                                or (layer.is_consumer and la.gdn_no_q_conv)
                            )
                        )
                        or (
                            layer.is_consumer
                            and GDN_CONSUMER_Q_CONV_TAG in layer_caches
                            and (yoco_on or not la.gdn_no_q_conv)
                        )
                    )
                ):
                    ssm_pool_2d = None
                    bt_ssm_op = None
                    tpb_ssm_op = 1
                    ssm_col_off = 0
                    conv_pool_2d = None
                    bt_conv_op = None
                    tpb_conv_op = 1
                    conv_col_off = 0
                    ssm_n = math.prod(self.ssm_shape)
                    if LINEAR_ATTN_TAG in layer_caches:
                        c_lin = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        ssm_pool_2d, _ = _view_opaque_pool_2d(c_lin.kv_cache_base, torch.float32)
                        conv_pool_2d, _ = _view_opaque_pool_2d(c_lin.kv_cache_base, dtype)
                        bt_ssm_op = bt_lin
                        bt_conv_op = bt_lin
                        tpb_ssm_op = int(c_lin.seq_size_per_block)
                        tpb_conv_op = tpb_ssm_op
                        conv_col_off = 2 * ssm_n
                    elif GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm_op = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_pool_2d, _ = _view_opaque_pool_2d(c_ssm.kv_cache_base, torch.float32)
                        tpb_ssm_op = int(c_ssm.seq_size_per_block)
                        c_conv_tag = GDN_CONV_STATE_TAG if GDN_CONV_STATE_TAG in layer_caches else (
                            GDN_CONSUMER_Q_CONV_TAG if (layer.is_consumer and GDN_CONSUMER_Q_CONV_TAG in layer_caches) else None
                        )
                        if c_conv_tag is not None:
                            c_cv = layer_caches[c_conv_tag]
                            bt_conv_op = _block_table_for_tag(attention_inputs, c_conv_tag)
                            conv_pool_2d, _ = _view_opaque_pool_2d(c_cv.kv_cache_base, dtype)
                            tpb_conv_op = int(c_cv.seq_size_per_block)
                    else:
                        c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                        bt_conv_op = _block_table_for_tag(attention_inputs, GDN_CONSUMER_Q_CONV_TAG)
                        conv_pool_2d, _ = _view_opaque_pool_2d(c_q.kv_cache_base, dtype)
                        tpb_conv_op = int(c_q.seq_size_per_block)
                    if not (layer.is_consumer and not yoco_on and GDN_SSM_STATE_TAG not in layer_caches and LINEAR_ATTN_TAG not in layer_caches):
                        attn_out, published = la.forward_decode_opaque(
                            attn_in,
                            conv_pool_2d=conv_pool_2d,
                            bt_conv=bt_conv_op,
                            tpb_conv=tpb_conv_op,
                            conv_col_offset=conv_col_off,
                            ssm_pool_2d=ssm_pool_2d,
                            bt_ssm=bt_ssm_op,
                            tpb_ssm=tpb_ssm_op,
                            ssm_col_offset=ssm_col_off,
                            seq_before=seq_lens_before_dev,
                            shared=(shared_gdn[layer.source_layer_idx] if layer.is_consumer else None),
                            enable_yoco_decode=yoco_on,
                            skip_allreduce=dec_sp_active,
                        )
                        if published is not None:
                            shared_gdn[lid] = published
                        can_gdn_opaque = True
                    else:
                        can_gdn_opaque = False
                else:
                    can_gdn_opaque = False
                if can_gdn_opaque:
                    pass
                elif layer.is_consumer:
                    q_state: Optional[GDNLayerState] = None
                    bt_q = None
                    if GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_buf = self._read_opaque_batch_gpu(
                            c_ssm,
                            bt_ssm,
                            seq_lens_before_dev,
                            GDN_SSM_STATE_TAG,
                            lid,
                            target_dtype=torch.float32,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_q = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            q_conv_buf = self._read_opaque_batch_gpu(
                                c_conv,
                                bt_q,
                                seq_lens_before_dev,
                                GDN_CONSUMER_Q_CONV_TAG,
                                lid,
                                target_dtype=dtype,
                            )
                        else:
                            q_conv_buf = torch.zeros(
                                (bsz, *self.gdn_consumer_conv_shape), dtype=dtype, device=device
                            )
                        q_state = GDNLayerState(conv_state=q_conv_buf, ssm_state=ssm_buf)
                    elif GDN_CONSUMER_Q_CONV_TAG in layer_caches:
                        c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                        bt_q = _block_table_for_tag(attention_inputs, GDN_CONSUMER_Q_CONV_TAG)
                        q_conv_buf = self._read_opaque_batch_gpu(
                            c_q,
                            bt_q,
                            seq_lens_before_dev,
                            GDN_CONSUMER_Q_CONV_TAG,
                            lid,
                            target_dtype=dtype,
                        )
                        q_state = GDNLayerState(
                            conv_state=q_conv_buf,
                            ssm_state=(
                                None
                                if yoco_on
                                else torch.zeros(
                                    (bsz, *self.ssm_shape), dtype=torch.float32, device=device
                                )
                            ),
                        )
                    elif LINEAR_ATTN_TAG in layer_caches:
                        c_lin = layer_caches[LINEAR_ATTN_TAG]
                        bt_q = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        packed = self._read_opaque_batch_gpu(
                            c_lin,
                            bt_q,
                            seq_lens_before_dev,
                            LINEAR_ATTN_TAG,
                            lid,
                            target_dtype=torch.float32,
                        )
                        ssm_n = math.prod(self.ssm_shape)
                        conv_shape = self.gdn_consumer_conv_shape
                        conv_n = math.prod(conv_shape)
                        ssm_buf = packed[:, :ssm_n].reshape(bsz, *self.ssm_shape).clone()
                        conv_buf = (
                            packed[:, ssm_n:]
                            .contiguous()
                            .view(torch.bfloat16)[:, :conv_n]
                            .reshape(bsz, *conv_shape)
                            .to(dtype)
                            .clone()
                        )
                        q_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                    else:
                        raise RuntimeError(
                            f"Unsupported linear consumer cache layout at layer {lid}: {list(layer_caches.keys())}"
                        )
                    shared_in = shared_gdn[layer.source_layer_idx]  # type: ignore[index]
                    attn_out, _ = layer.linear_attn(
                        attn_in,
                        state=q_state,
                        shared=shared_in,
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=dec_sp_active,
                    )
                    if q_state is not None and LINEAR_ATTN_TAG in layer_caches and bt_q is not None:
                        packed_out = torch.zeros_like(packed)
                        packed_out[:, :ssm_n].copy_(
                            q_state.ssm_state.reshape(bsz, ssm_n).float()
                        )
                        packed_out[:, ssm_n:].view(torch.bfloat16)[:, :conv_n].copy_(
                            q_state.conv_state.reshape(bsz, conv_n).to(torch.bfloat16)
                        )
                        self._write_opaque_batch_gpu(
                            layer_caches[LINEAR_ATTN_TAG],
                            bt_q,
                            seq_lens_before_dev,
                            packed_out,
                            LINEAR_ATTN_TAG,
                            lid,
                        )
                    if q_state is not None and GDN_SSM_STATE_TAG in layer_caches:
                        assert q_state.ssm_state is not None
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        self._write_opaque_batch_gpu(
                            layer_caches[GDN_SSM_STATE_TAG],
                            bt_ssm,
                            seq_lens_before_dev,
                            q_state.ssm_state,
                            GDN_SSM_STATE_TAG,
                            lid,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches and bt_q is not None:
                            self._write_opaque_batch_gpu(
                                layer_caches[GDN_CONV_STATE_TAG],
                                bt_q,
                                seq_lens_before_dev,
                                q_state.conv_state,
                                GDN_CONV_STATE_TAG,
                                lid,
                            )
                    elif q_state is not None and GDN_CONSUMER_Q_CONV_TAG in layer_caches and bt_q is not None:
                        self._write_opaque_batch_gpu(
                            layer_caches[GDN_CONSUMER_Q_CONV_TAG],
                            bt_q,
                            seq_lens_before_dev,
                            q_state.conv_state,
                            GDN_CONSUMER_Q_CONV_TAG,
                            lid,
                        )
                elif LINEAR_ATTN_TAG in layer_caches:
                    c_lin = layer_caches[LINEAR_ATTN_TAG]
                    bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                    packed = self._read_opaque_batch_gpu(
                        c_lin,
                        bt_lin,
                        seq_lens_before_dev,
                        LINEAR_ATTN_TAG,
                        lid,
                        target_dtype=torch.float32,
                    )
                    ssm_n = math.prod(self.ssm_shape)
                    conv_n = math.prod(self.gdn_conv_shape)
                    ssm_buf = packed[:, :ssm_n].reshape(bsz, *self.ssm_shape).clone()
                    conv_buf = (
                        packed[:, ssm_n:]
                        .contiguous()
                        .view(torch.bfloat16)[:, :conv_n]
                        .reshape(bsz, *self.gdn_conv_shape)
                        .to(dtype)
                        .clone()
                    )
                    gdn_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                    attn_out, published = layer.linear_attn(
                        attn_in,
                        state=gdn_state,
                        shared=None,
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=dec_sp_active,
                    )
                    if published is not None:
                        shared_gdn[lid] = published
                    assert gdn_state.ssm_state is not None
                    packed_out = torch.zeros_like(packed)
                    packed_out[:, :ssm_n].copy_(
                        gdn_state.ssm_state.reshape(bsz, ssm_n).float()
                    )
                    packed_out[:, ssm_n:].view(torch.bfloat16)[:, :conv_n].copy_(
                        gdn_state.conv_state.reshape(bsz, conv_n).to(torch.bfloat16)
                    )
                    self._write_opaque_batch_gpu(
                        c_lin,
                        bt_lin,
                        seq_lens_before_dev,
                        packed_out,
                        LINEAR_ATTN_TAG,
                        lid,
                    )
                else:
                    if GDN_SSM_STATE_TAG not in layer_caches:
                        raise RuntimeError(
                            f"Unsupported linear cache layout at layer {lid}: {list(layer_caches.keys())}"
                        )
                    c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                    bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                    ssm_buf = self._read_opaque_batch_gpu(
                        c_ssm,
                        bt_ssm,
                        seq_lens_before_dev,
                        GDN_SSM_STATE_TAG,
                        lid,
                        target_dtype=torch.float32,
                    )
                    if GDN_CONV_STATE_TAG in layer_caches:
                        c_conv = layer_caches[GDN_CONV_STATE_TAG]
                        bt_conv = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                        conv_buf = self._read_opaque_batch_gpu(
                            c_conv,
                            bt_conv,
                            seq_lens_before_dev,
                            GDN_CONV_STATE_TAG,
                            lid,
                            target_dtype=dtype,
                        )
                    else:
                        bt_conv = None
                        conv_buf = torch.zeros(
                            (bsz, *self.gdn_conv_shape), dtype=dtype, device=device
                        )
                    gdn_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                    attn_out, published = layer.linear_attn(
                        attn_in,
                        state=gdn_state,
                        shared=None,
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=dec_sp_active,
                    )
                    if published is not None:
                        shared_gdn[lid] = published
                    assert gdn_state.ssm_state is not None
                    self._write_opaque_batch_gpu(
                        c_ssm,
                        bt_ssm,
                        seq_lens_before_dev,
                        gdn_state.ssm_state,
                        GDN_SSM_STATE_TAG,
                        lid,
                    )
                    if GDN_CONV_STATE_TAG in layer_caches and bt_conv is not None:
                        self._write_opaque_batch_gpu(
                            layer_caches[GDN_CONV_STATE_TAG],
                            bt_conv,
                            seq_lens_before_dev,
                            gdn_state.conv_state,
                            GDN_CONV_STATE_TAG,
                            lid,
                        )
            else:
                assert layer.self_attn is not None
                sa = layer.self_attn
                preproj = sa.project_qkv_and_indexer(
                    attn_in, need_idx=(sa.indexer is not None), need_idx_norm=False
                )
                if not layer.is_consumer:
                    kv_c = layer_caches[FULL_ATTN_KV_TAG]
                    sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                    bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                    k_sc_st: Optional[ShortConvState] = None
                    v_sc_st: Optional[ShortConvState] = None
                    bt_sc = None
                    used_preconv_kv = False
                    if sc_c is not None:
                        bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
                        if (
                            sa.use_kv_sconv
                            and sa.k_sconv is not None
                            and sa.v_sconv is not None
                            and preproj[1] is not None
                            and preproj[2] is not None
                            and attn_in.is_cuda
                            and dtype == torch.bfloat16
                            and sa.k_sconv.kernel_size == 4
                            and sa.v_sconv.kernel_size == 4
                        ):
                            sc_pool_bf16, _ = _view_opaque_pool_2d(sc_c.kv_cache_base, dtype)
                            tpb_kv_sc = int(sc_c.seq_size_per_block)
                            kv_dim_sc = self.cfg.num_key_value_heads * self.cfg.head_dim
                            k_conv = causal_dwconv1d_decode_opaque_triton(
                                preproj[1],
                                sa.k_sconv.weight,
                                sc_pool_bf16,
                                bt_sc,
                                seq_lens_before_dev,
                                page_size=tpb_kv_sc,
                                pool_col_offset=0,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                            )
                            v_conv = causal_dwconv1d_decode_opaque_triton(
                                preproj[2],
                                sa.v_sconv.weight,
                                sc_pool_bf16,
                                bt_sc,
                                seq_lens_before_dev,
                                page_size=tpb_kv_sc,
                                pool_col_offset=kv_dim_sc * 3,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                            )
                            preproj = (preproj[0], k_conv, v_conv, preproj[3], preproj[4])
                            used_preconv_kv = True
                        else:
                            kv_sc_buf = self._read_opaque_batch_gpu(
                                sc_c,
                                bt_sc,
                                seq_lens_before_dev,
                                FULL_ATTN_KV_SCONV_TAG,
                                lid,
                                target_dtype=dtype,
                            )
                            k_sc_st = ShortConvState(kv_sc_buf[:, 0].clone())
                            v_sc_st = ShortConvState(kv_sc_buf[:, 1].clone())

                    if layer.self_attn.indexer is not None:
                        if not isinstance(decode_attn_handle, Qwen4_5QSADecodeHandle):
                            raise RuntimeError(
                                f"QSA layer {lid} decode requires Qwen4_5QSADecodeHandle"
                            )
                        idx = layer.self_attn.indexer
                        idx_kv_c = layer_caches[INDEXER_KV_TAG]
                        idx_st_c = layer_caches[INDEXER_STATE_TAG]
                        bt_idx_kv = _block_table_for_tag(attention_inputs, INDEXER_KV_TAG)
                        bt_idx_st = _block_table_for_tag(attention_inputs, INDEXER_STATE_TAG)
                        idx_kv_tpb = int(idx_kv_c.seq_size_per_block)
                        idx_st_tpb = int(idx_st_c.seq_size_per_block)
                        idx_kv_pool = view_indexer_kv_pool(
                            idx_kv_c,
                            kv_tokens_per_block=idx_kv_tpb,
                            compress_ratio=idx.compress_ratio,
                            head_dim=idx.head_dim,
                        )
                        idx_st_pool = view_indexer_state_pool(
                            idx_st_c,
                            compress_ratio=idx.compress_ratio,
                            head_dim=idx.head_dim,
                        )
                        assert preproj[3] is not None and preproj[4] is not None
                        q_idx, raw_k_idx = preproj[3], preproj[4]
                        use_pad_mask = decode_attn_handle.is_cuda_graph or _is_padded_group(
                            attention_inputs, FULL_ATTN_KV_TAG
                        )
                        active_rows, visible_lens = _get_qsa_vis_dec(bt_idx_st, use_pad_mask)
                        write_qsa_decode_cache_gpu_(
                            raw_k_idx.squeeze(1),
                            seq_lens_before_dev,
                            idx.k_norm_gamma,
                            idx_kv_pool,
                            bt_idx_kv,
                            idx_st_pool,
                            bt_idx_st,
                            kv_tokens_per_block=idx_kv_tpb,
                            state_tokens_per_block=idx_st_tpb,
                            ratio=idx.compress_ratio,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            norm_eps=idx.norm_eps,
                            valid=active_rows,
                        )
                        selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            idx_kv_pool,
                            bt_idx_kv,
                            visible_lens,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            max_ctx_len=decode_qsa_max_ctx_len,
                            is_cuda_graph=decode_attn_handle.is_cuda_graph,
                            q_norm_gamma=idx.q_norm_gamma,
                            norm_eps=idx.norm_eps,
                            **cp_idx_kw,
                        )
                        source_idx_caches[lid] = (idx_kv_pool, bt_idx_kv)

                        def _qsa_src_decode_fn(
                            q_3d: torch.Tensor,
                            kv_l: Any,
                            _bt: torch.Tensor = bt_kv,
                            _vl: torch.Tensor = visible_lens,
                            _sel: torch.Tensor = selected,
                        ) -> torch.Tensor:
                            return decode_attn_handle.forward(
                                q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, selected=_sel
                            )

                        decode_fn = _qsa_src_decode_fn
                    else:
                        decode_fn = decode_attn_handle.forward

                    attn_out = sa.forward_paged_decode(
                        attn_in,
                        positions_2d,
                        kv_c,
                        decode_fn,
                        block_table=bt_kv,
                        seq_before_dev=seq_lens_before_dev,
                        k_sconv_state=k_sc_st,
                        v_sconv_state=v_sc_st,
                        write_kv_fn=_store_decode_full_attn_kv_gpu,
                        preprojected=preproj,
                        preconv_kv=used_preconv_kv,
                        skip_allreduce=dec_sp_active,
                    )
                    full_kv_layer_caches[lid] = kv_c
                    if (
                        not used_preconv_kv
                        and sc_c is not None
                        and bt_sc is not None
                        and k_sc_st is not None
                        and v_sc_st is not None
                    ):
                        packed_sc = torch.stack(
                            [k_sc_st.conv_state, v_sc_st.conv_state], dim=1
                        )
                        self._write_opaque_batch_gpu(
                            sc_c,
                            bt_sc,
                            seq_lens_before_dev,
                            packed_sc,
                            FULL_ATTN_KV_SCONV_TAG,
                            lid,
                        )
                else:
                    src_kv_c = full_kv_layer_caches[layer.kv_cache_owner]
                    if layer.self_attn.indexer is not None:
                        if not isinstance(decode_attn_handle, Qwen4_5QSADecodeHandle):
                            raise RuntimeError(
                                f"QSA consumer layer {lid} decode requires Qwen4_5QSADecodeHandle"
                            )
                        idx = layer.self_attn.indexer
                        src_idx_kv_pool, src_bt_idx_kv = source_idx_caches[
                            layer.kv_cache_owner
                        ]
                        src_bt_kv = _block_table_for_tag(
                            attention_inputs, FULL_ATTN_KV_TAG
                        )
                        assert preproj[3] is not None
                        q_idx = preproj[3]
                        use_pad_mask = decode_attn_handle.is_cuda_graph or _is_padded_group(
                            attention_inputs, FULL_ATTN_KV_TAG
                        )
                        _, visible_lens = _get_qsa_vis_dec(src_bt_idx_kv, use_pad_mask)
                        selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            src_idx_kv_pool,
                            src_bt_idx_kv,
                            visible_lens,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            max_ctx_len=decode_qsa_max_ctx_len,
                            is_cuda_graph=decode_attn_handle.is_cuda_graph,
                            q_norm_gamma=idx.q_norm_gamma,
                            norm_eps=idx.norm_eps,
                            **cp_idx_kw,
                        )

                        def _qsa_cons_decode_fn(
                            q_3d: torch.Tensor,
                            kv_l: Any,
                            _bt: torch.Tensor = src_bt_kv,
                            _vl: torch.Tensor = visible_lens,
                            _sel: torch.Tensor = selected,
                        ) -> torch.Tensor:
                            return decode_attn_handle.forward(
                                q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, selected=_sel
                            )

                        c_decode_fn = _qsa_cons_decode_fn
                    else:
                        c_decode_fn = decode_attn_handle.forward

                    attn_out = sa.forward_paged_decode(
                        attn_in,
                        positions_2d,
                        src_kv_c,
                        c_decode_fn,
                        preprojected=preproj,
                        skip_allreduce=dec_sp_active,
                    )

            if dec_sp_active:
                attn_out = self._sp_reduce_scatter_dim0(
                    attn_out.reshape(bsz, self.cfg.hidden_size),
                    total_tokens=bsz,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                ).view(b_loc, 1, self.cfg.hidden_size)

            # 4. Attention Output ShortConv + 5. MLP Hyper-Connection (fused opaque when K=4)
            if dec_sp_active and layer.attn_output_sconv is not None:
                tag_ao_s = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao_s = layer_caches[tag_ao_s]
                bt_ao_s = _block_table_for_tag(attention_inputs, tag_ao_s)
                ao_b = self.cfg.hidden_size * max(0, int(layer.attn_output_sconv.kernel_size) - 1) * _dtype_itemsize(dtype)
                if ao_b > 0:
                    dec_sp_sync_specs.append((c_ao_s, bt_ao_s, ao_b))
            mlp_hc = layer.mlp_hyper_connection
            if (
                layer.attn_output_sconv is not None
                and attn_out.is_cuda
                and dtype == torch.bfloat16
                and layer.attn_output_sconv.kernel_size == 4
            ):
                tag_out = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao = layer_caches[tag_out]
                bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                bt_ao_loc = bt_ao[b0:b1] if dec_sp_active else bt_ao
                ao_pool_bf16, _ = _view_opaque_pool_2d(c_ao.kv_cache_base, dtype)
                if mlp_hc.use_combine:
                    hyper_hidden, mlp_normed = hc_sconv_decode_opaque_combine_norm_triton(
                        attn_out,
                        hyper_hidden,
                        attn_inj,
                        mlp_hc.hc_norm_weight,
                        mlp_hc.hc_norm_bias,
                        layer.attn_output_sconv.weight,
                        ao_pool_bf16,
                        bt_ao_loc,
                        seq_loc_dev,
                        page_size=int(c_ao.seq_size_per_block),
                        eps=mlp_hc.eps,
                        conv_residual=layer.attn_output_sconv.residual,
                        conv_use_silu=layer.attn_output_sconv.use_activation,
                        clamp_val=self.cfg.clip_attn_output,
                    )
                    mlp_in, hyper_hidden, mlp_inj, prev_latent = mlp_hc.mix_from_normed(
                        mlp_normed, hyper_hidden, prev_latent
                    )
                else:
                    attn_out = causal_dwconv1d_decode_opaque_triton(
                        attn_out,
                        layer.attn_output_sconv.weight,
                        ao_pool_bf16,
                        bt_ao_loc,
                        seq_loc_dev,
                        page_size=int(c_ao.seq_size_per_block),
                        pool_col_offset=0,
                        residual=layer.attn_output_sconv.residual,
                        use_activation=layer.attn_output_sconv.use_activation,
                        clamp_val=self.cfg.clip_attn_output,
                    )
                    mlp_in, hyper_hidden, mlp_inj, prev_latent = mlp_hc.combine_and_mix_with_latent(
                        attn_out, hyper_hidden, attn_inj, prev_latent
                    )
            else:
                if layer.attn_output_sconv is not None:
                    tag_out = (
                        LINEAR_ATTN_OUTPUT_SCONV_TAG
                        if layer.layer_type == "linear_attention"
                        else FULL_ATTN_OUTPUT_SCONV_TAG
                    )
                    c_ao = layer_caches[tag_out]
                    bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                    bt_ao_loc = bt_ao[b0:b1] if dec_sp_active else bt_ao
                    ao_state = ShortConvState(
                        self._read_opaque_batch_gpu(
                            c_ao, bt_ao_loc, seq_loc_dev, tag_out, lid, target_dtype=dtype,
                        )
                    )
                    if attn_out.is_cuda and layer.attn_output_sconv.kernel_size == 4:
                        attn_out = causal_dwconv1d_decode_triton(
                            attn_out,
                            layer.attn_output_sconv.weight,
                            ao_state.conv_state,
                            residual=layer.attn_output_sconv.residual,
                            use_activation=layer.attn_output_sconv.use_activation,
                            clamp_val=self.cfg.clip_attn_output,
                        )
                    else:
                        attn_out = layer.attn_output_sconv(attn_out, state=ao_state)
                        if self.cfg.clip_attn_output is not None:
                            attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)
                    self._write_opaque_batch_gpu(
                        c_ao, bt_ao_loc, seq_loc_dev, ao_state.conv_state, tag_out, lid,
                    )
                elif self.cfg.clip_attn_output is not None:
                    attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)
                mlp_in, hyper_hidden, mlp_inj, prev_latent = mlp_hc.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )

            assert mlp_inj is not None
            mlp_out = _run_runtime_mlp(
                layer,
                mlp_in,
                sp_input=dec_sp_active,
                sp_total_tokens=bsz if dec_sp_active else None,
            )

            # 6. FFN Output ShortConv + clamp (defer K=4 CUDA to fuse with next layer's HC combine+norm)
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                bt_fo_loc = bt_fo[b0:b1] if dec_sp_active else bt_fo
                if dec_sp_active:
                    fo_b = self.cfg.hidden_size * max(0, int(layer.ffn_output_sconv.kernel_size) - 1) * _dtype_itemsize(dtype)
                    if fo_b > 0:
                        dec_sp_sync_specs.append((c_fo, bt_fo, fo_b))
                if mlp_out.is_cuda and dtype == torch.bfloat16 and layer.ffn_output_sconv.kernel_size == 4:
                    deferred_sconv = (
                        mlp_out,
                        mlp_inj,
                        c_fo,
                        bt_fo_loc,
                        layer.ffn_output_sconv,
                        self.cfg.clip_mlp_output,
                    )
                    deferred = None
                else:
                    fo_state = ShortConvState(
                        self._read_opaque_batch_gpu(
                            c_fo, bt_fo_loc, seq_loc_dev, FFN_OUTPUT_SCONV_TAG, lid, target_dtype=dtype,
                        )
                    )
                    if mlp_out.is_cuda and layer.ffn_output_sconv.kernel_size == 4:
                        mlp_out = causal_dwconv1d_decode_triton(
                            mlp_out,
                            layer.ffn_output_sconv.weight,
                            fo_state.conv_state,
                            residual=layer.ffn_output_sconv.residual,
                            use_activation=layer.ffn_output_sconv.use_activation,
                            clamp_val=self.cfg.clip_mlp_output,
                        )
                    else:
                        mlp_out = layer.ffn_output_sconv(mlp_out, state=fo_state)
                        if self.cfg.clip_mlp_output is not None:
                            mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)
                    self._write_opaque_batch_gpu(
                        c_fo, bt_fo_loc, seq_loc_dev, fo_state.conv_state, FFN_OUTPUT_SCONV_TAG, lid,
                    )
                    deferred = (mlp_out, mlp_inj)
            else:
                if self.cfg.clip_mlp_output is not None:
                    mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)
                deferred = (mlp_out, mlp_inj)

        _flush_deferred_sconv()
        if dec_sp_active and dec_sp_sync_specs:
            self._sync_sp_decode_opaque_states(
                dec_sp_sync_specs,
                seq_lens_before_dev,
                bsz=bsz,
                b0=b0,
                b1=b1,
                sp_size=eff_sp,
                sp_rank=eff_rk,
                sp_group=eff_grp,
                sp_comm=eff_comm,
                device=device,
            )
        assert deferred is not None
        hidden, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
            deferred[0], hyper_hidden, deferred[1], prev_latent=None
        )
        hidden_out = hidden.reshape(b_loc, self.cfg.hidden_size)
        if dec_sp_active:
            hidden_out = self._sp_all_gather_dim0(
                hidden_out,
                total_tokens=bsz,
                sp_size=eff_sp,
                sp_rank=eff_rk,
                sp_group=eff_grp,
                sp_comm=eff_comm,
            )
            if aux_out is not None:
                aux_out = self._sp_all_gather_dim0(
                    aux_out,
                    total_tokens=bsz,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                )
        if aux_out is not None:
            return hidden_out, aux_out
        return hidden_out

    def run_decode_batch(
        self,
        model: Qwen4_5Model,
        input_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
        writers: Mapping[str, Any],
        *,
        seq_lens_before: Optional[List[int]] = None,
        seq_lens_before_dev: Optional[torch.Tensor] = None,
        decode_attn_handle: Optional[Any] = None,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Vectorized Decode path ([B, 1]) backed directly by paged ``KVCache`` pools."""
        with _model_a2a_step_scope(model):
            return self._run_decode_batch_impl(
                model,
                input_ids_2d,
                positions_2d,
                kv_cache,
                attention_inputs,
                writers,
                seq_lens_before=seq_lens_before,
                seq_lens_before_dev=seq_lens_before_dev,
                decode_attn_handle=decode_attn_handle,
                enable_yoco_decode=enable_yoco_decode,
                capture_layer_ids=capture_layer_ids,
            )

    def _run_decode_batch_impl(
        self,
        model: Qwen4_5Model,
        input_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
        writers: Mapping[str, Any],
        *,
        seq_lens_before: Optional[List[int]] = None,
        seq_lens_before_dev: Optional[torch.Tensor] = None,
        decode_attn_handle: Optional[Any] = None,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if (
            seq_lens_before_dev is not None
            and isinstance(
                decode_attn_handle, (Qwen4_5DecodeAttnHandle, Qwen4_5QSADecodeHandle)
            )
            and input_ids_2d.device.type == "cuda"
            and _HAS_TRITON
        ):
            return self._run_decode_batch_gpu(
                model,
                input_ids_2d,
                positions_2d,
                kv_cache,
                attention_inputs,
                seq_lens_before_dev=seq_lens_before_dev,
                decode_attn_handle=decode_attn_handle,
                enable_yoco_decode=enable_yoco_decode,
                capture_layer_ids=capture_layer_ids,
            )
        assert seq_lens_before is not None
        device = input_ids_2d.device
        dtype = model.embed_tokens.weight.dtype
        bsz = int(input_ids_2d.shape[0])
        if bsz == 0:
            self._run_empty_ep_collectives(model, dtype, device)
            empty_h = torch.empty((0, self.cfg.hidden_size), dtype=dtype, device=device)
            if capture_layer_ids:
                return empty_h, torch.empty((0, len(capture_layer_ids) * self.cfg.hidden_size), dtype=dtype, device=device)
            return empty_h
        new_lens = [s + 1 for s in seq_lens_before]
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )

        # Precompute FullAttn block tables and read paged KV caches per producer layer
        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None
        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = torch.empty(
                (bsz, len(capture_layer_ids) * self.cfg.hidden_size),
                dtype=dtype,
                device=device,
            )
        full_block_table = (
            _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
            if kv_cache is not None
            else None
        )
        is_padded = _is_padded_group(attention_inputs, FULL_ATTN_KV_TAG)
        max_prev = max(seq_lens_before) if seq_lens_before else 0

        emb = model.embed_tokens(input_ids_2d)
        hyper_hidden = emb.repeat(1, 1, self.cfg.hc_count)
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn: Dict[int, Any] = {}
        full_kv_ephemeral: Dict[int, FullAttnKVCache] = {}
        source_idx_local: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        cp_idx_kw = self._cp_indexer_kwargs()

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}

            # 1. Optional PLE
            if layer.ple is not None:
                if deferred is not None:
                    hyper_hidden = layer.attn_hyper_connection.combine(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                    deferred = None
                ple_st_cache = layer_caches[PLE_STATE_TAG]
                ple_ctx_cache = layer_caches[PLE_NGRAM_CTX_TAG]
                bt_st = _block_table_for_tag(attention_inputs, PLE_STATE_TAG)
                bt_ctx = _block_table_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
                ple_state = PLELayerState(
                    ngram_context=self._read_opaque_batch(
                        ple_ctx_cache,
                        bt_ctx,
                        seq_lens_before,
                        PLE_NGRAM_CTX_TAG,
                        lid,
                        fill_value=self.cfg.eos_token_id,
                        allow_zero_padded=is_padded,
                    ),
                    conv_state=self._read_opaque_batch(
                        ple_st_cache,
                        bt_st,
                        seq_lens_before,
                        PLE_STATE_TAG,
                        lid,
                        target_dtype=dtype,
                        allow_zero_padded=is_padded,
                    ),
                )
                hyper_hidden = layer.ple(hyper_hidden, input_ids_2d, state=ple_state)
                self._write_opaque_batch(
                    ple_st_cache,
                    bt_st,
                    new_lens,
                    ple_state.conv_state,
                    PLE_STATE_TAG,
                    lid,
                    allow_zero_padded=is_padded,
                )
                self._write_opaque_batch(
                    ple_ctx_cache,
                    bt_ctx,
                    new_lens,
                    ple_state.ngram_context,
                    PLE_NGRAM_CTX_TAG,
                    lid,
                    allow_zero_padded=is_padded,
                )

            # 2. Attn Hyper-Connection
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.mix_with_latent(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.combine_and_mix_with_latent(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
            assert attn_inj is not None
            if aux_out is not None and lid in cap_slots:
                attn_flat = attn_in.reshape(bsz, self.cfg.hidden_size)
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * self.cfg.hidden_size : (slot_i + 1) * self.cfg.hidden_size].copy_(
                        attn_flat
                    )

            # 3. Attention Core (GDN vs FullAttn)
            if layer.linear_attn is not None:
                if layer.is_consumer:
                    q_state: Optional[GDNLayerState] = None
                    bt_q = None
                    bt_lin_c = None
                    base_lin_c = None
                    tpb_lin_c = 0
                    if LINEAR_ATTN_TAG in layer_caches:
                        lin_cache = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin_c = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        tpb_lin_c = int(lin_cache.seq_size_per_block)
                        base_lin_c = lin_cache.kv_cache_base
                        q_conv_buf = torch.zeros(
                            (bsz, *self.gdn_consumer_conv_shape), dtype=dtype, device=device
                        )
                        ssm_buf = torch.zeros(
                            (bsz, *self.ssm_shape), dtype=torch.float32, device=device
                        )
                        for b, plen in enumerate(seq_lens_before):
                            if plen > 0:
                                page = (plen - 1) // tpb_lin_c
                                phys = _check_phys_block(
                                    int(bt_lin_c[b, page].item()),
                                    int(base_lin_c.shape[0]),
                                    LINEAR_ATTN_TAG,
                                    lid,
                                )
                                ssm_b, conv_b = _unpack_linear_block(
                                    base_lin_c[phys],
                                    self.ssm_shape,
                                    self.gdn_consumer_conv_shape,
                                    dtype,
                                )
                                ssm_buf[b].copy_(ssm_b)
                                q_conv_buf[b].copy_(conv_b)
                        q_state = GDNLayerState(conv_state=q_conv_buf, ssm_state=ssm_buf)
                    elif GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_buf = self._read_opaque_batch(
                            c_ssm,
                            bt_ssm,
                            seq_lens_before,
                            GDN_SSM_STATE_TAG,
                            lid,
                            target_dtype=torch.float32,
                            allow_zero_padded=is_padded,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_q = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            q_conv_buf = self._read_opaque_batch(
                                c_conv,
                                bt_q,
                                seq_lens_before,
                                GDN_CONSUMER_Q_CONV_TAG,
                                lid,
                                target_dtype=dtype,
                                allow_zero_padded=is_padded,
                            )
                        else:
                            q_conv_buf = torch.zeros(
                                (bsz, *self.gdn_consumer_conv_shape), dtype=dtype, device=device
                            )
                        q_state = GDNLayerState(conv_state=q_conv_buf, ssm_state=ssm_buf)
                    elif GDN_CONSUMER_Q_CONV_TAG in layer_caches:
                        c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                        bt_q = _block_table_for_tag(attention_inputs, GDN_CONSUMER_Q_CONV_TAG)
                        q_conv_buf = self._read_opaque_batch(
                            c_q,
                            bt_q,
                            seq_lens_before,
                            GDN_CONSUMER_Q_CONV_TAG,
                            lid,
                            target_dtype=dtype,
                            allow_zero_padded=is_padded,
                        )
                        q_state = GDNLayerState(
                            conv_state=q_conv_buf,
                            ssm_state=(
                                None
                                if yoco_on
                                else torch.zeros(
                                    (bsz, *self.ssm_shape), dtype=torch.float32, device=device
                                )
                            ),
                        )
                    shared_in = shared_gdn[layer.source_layer_idx]  # type: ignore[index]
                    attn_out, _ = layer.linear_attn(
                        attn_in,
                        state=q_state,
                        shared=shared_in,
                        enable_yoco_decode=yoco_on,
                    )
                    if q_state is not None and LINEAR_ATTN_TAG in layer_caches and bt_lin_c is not None and base_lin_c is not None:
                        assert q_state.ssm_state is not None
                        for b, nlen in enumerate(new_lens):
                            page = (nlen - 1) // tpb_lin_c
                            raw_phys = int(bt_lin_c[b, page].item())
                            if is_padded and raw_phys == 0:
                                continue
                            phys = _check_phys_block(
                                raw_phys,
                                int(base_lin_c.shape[0]),
                                LINEAR_ATTN_TAG,
                                lid,
                            )
                            _pack_linear_block(
                                base_lin_c[phys], q_state.ssm_state[b], q_state.conv_state[b]
                            )
                    elif q_state is not None and GDN_SSM_STATE_TAG in layer_caches:
                        assert q_state.ssm_state is not None
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        self._write_opaque_batch(
                            layer_caches[GDN_SSM_STATE_TAG],
                            bt_ssm,
                            new_lens,
                            q_state.ssm_state,
                            GDN_SSM_STATE_TAG,
                            lid,
                            allow_zero_padded=is_padded,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches and bt_q is not None:
                            self._write_opaque_batch(
                                layer_caches[GDN_CONV_STATE_TAG],
                                bt_q,
                                new_lens,
                                q_state.conv_state,
                                GDN_CONV_STATE_TAG,
                                lid,
                                allow_zero_padded=is_padded,
                            )
                    elif q_state is not None and GDN_CONSUMER_Q_CONV_TAG in layer_caches and bt_q is not None:
                        self._write_opaque_batch(
                            layer_caches[GDN_CONSUMER_Q_CONV_TAG],
                            bt_q,
                            new_lens,
                            q_state.conv_state,
                            GDN_CONSUMER_Q_CONV_TAG,
                            lid,
                            allow_zero_padded=is_padded,
                        )
                else:
                    if GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_buf = self._read_opaque_batch(
                            c_ssm,
                            bt_ssm,
                            seq_lens_before,
                            GDN_SSM_STATE_TAG,
                            lid,
                            target_dtype=torch.float32,
                            allow_zero_padded=is_padded,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_conv = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            conv_buf = self._read_opaque_batch(
                                c_conv,
                                bt_conv,
                                seq_lens_before,
                                GDN_CONV_STATE_TAG,
                                lid,
                                target_dtype=dtype,
                                allow_zero_padded=is_padded,
                            )
                        else:
                            bt_conv = None
                            conv_buf = torch.zeros((bsz, *self.gdn_conv_shape), dtype=dtype, device=device)
                        gdn_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                        attn_out, published = layer.linear_attn(
                            attn_in,
                            state=gdn_state,
                            shared=None,
                            enable_yoco_decode=yoco_on,
                        )
                        if published is not None:
                            shared_gdn[lid] = published
                        assert gdn_state.ssm_state is not None
                        self._write_opaque_batch(
                            c_ssm,
                            bt_ssm,
                            new_lens,
                            gdn_state.ssm_state,
                            GDN_SSM_STATE_TAG,
                            lid,
                            allow_zero_padded=is_padded,
                        )
                        if GDN_CONV_STATE_TAG in layer_caches and bt_conv is not None:
                            self._write_opaque_batch(
                                layer_caches[GDN_CONV_STATE_TAG],
                                bt_conv,
                                new_lens,
                                gdn_state.conv_state,
                                GDN_CONV_STATE_TAG,
                                lid,
                                allow_zero_padded=is_padded,
                            )
                    else:
                        lin_cache = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        tpb = int(lin_cache.seq_size_per_block)
                        base_lin = lin_cache.kv_cache_base
                        conv_buf = torch.zeros((bsz, *self.gdn_conv_shape), dtype=dtype, device=device)
                        ssm_buf = torch.zeros((bsz, *self.ssm_shape), dtype=torch.float32, device=device)
                        for b, plen in enumerate(seq_lens_before):
                            if plen > 0:
                                page = (plen - 1) // tpb
                                raw_phys = int(bt_lin[b, page].item())
                                if is_padded and raw_phys == 0:
                                    continue
                                phys = _check_phys_block(raw_phys, int(base_lin.shape[0]), LINEAR_ATTN_TAG, lid)
                                ssm_b, conv_b = _unpack_linear_block(
                                    base_lin[phys], self.ssm_shape, self.gdn_conv_shape, dtype
                                )
                                ssm_buf[b].copy_(ssm_b)
                                conv_buf[b].copy_(conv_b)
                        gdn_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                        attn_out, published = layer.linear_attn(
                            attn_in,
                            state=gdn_state,
                            shared=None,
                            enable_yoco_decode=yoco_on,
                        )
                        if published is not None:
                            shared_gdn[lid] = published
                        for b, nlen in enumerate(new_lens):
                            page = (nlen - 1) // tpb
                            raw_phys = int(bt_lin[b, page].item())
                            if is_padded and raw_phys == 0:
                                continue
                            phys = _check_phys_block(raw_phys, int(base_lin.shape[0]), LINEAR_ATTN_TAG, lid)
                            _pack_linear_block(base_lin[phys], gdn_state.ssm_state[b], gdn_state.conv_state[b])
            else:
                assert layer.self_attn is not None and full_block_table is not None
                sa = layer.self_attn
                preproj = sa.project_qkv_and_indexer(
                    attn_in, need_idx=(sa.indexer is not None)
                )
                if not layer.is_consumer:
                    kv_c = layer_caches[FULL_ATTN_KV_TAG]
                    sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                    bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                    ephemeral_kv = self._unpack_decode_kv_to_padded(
                        kv_c,
                        bt_kv,
                        seq_lens_before,
                        max_prev,
                        device,
                        dtype,
                        allow_zero_padded=is_padded,
                    )
                    k_sc_st: Optional[ShortConvState] = None
                    v_sc_st: Optional[ShortConvState] = None
                    bt_sc = None
                    if sc_c is not None:
                        bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
                        kv_sc_buf = self._read_opaque_batch(
                            sc_c,
                            bt_sc,
                            seq_lens_before,
                            FULL_ATTN_KV_SCONV_TAG,
                            lid,
                            target_dtype=dtype,
                            allow_zero_padded=is_padded,
                        )
                        k_sc_st = ShortConvState(kv_sc_buf[:, 0].clone())
                        v_sc_st = ShortConvState(kv_sc_buf[:, 1].clone())

                    src_selected: Optional[torch.Tensor] = None
                    if layer.self_attn.indexer is not None:
                        idx = layer.self_attn.indexer
                        idx_kv_c = layer_caches[INDEXER_KV_TAG]
                        idx_st_c = layer_caches[INDEXER_STATE_TAG]
                        bt_idx_kv = _block_table_for_tag(attention_inputs, INDEXER_KV_TAG)
                        bt_idx_st = _block_table_for_tag(attention_inputs, INDEXER_STATE_TAG)
                        idx_kv_tpb = int(idx_kv_c.seq_size_per_block)
                        idx_st_tpb = int(idx_st_c.seq_size_per_block)
                        idx_kv_pool = view_indexer_kv_pool(
                            idx_kv_c,
                            kv_tokens_per_block=idx_kv_tpb,
                            compress_ratio=idx.compress_ratio,
                            head_dim=idx.head_dim,
                        )
                        idx_st_pool = view_indexer_state_pool(
                            idx_st_c,
                            compress_ratio=idx.compress_ratio,
                            head_dim=idx.head_dim,
                        )
                        assert preproj[3] is not None and preproj[4] is not None
                        q_idx, raw_k_idx = preproj[3], preproj[4]
                        seq_b_t = torch.tensor(
                            seq_lens_before, dtype=torch.int32, device=device
                        )
                        active_rows = (
                            (bt_idx_st != 0).any(dim=1) if is_padded else None
                        )
                        vis_t = (
                            torch.where(active_rows, seq_b_t + 1, 0)
                            if active_rows is not None
                            else (seq_b_t + 1)
                        )
                        write_qsa_decode_cache_gpu_(
                            raw_k_idx.squeeze(1),
                            seq_b_t,
                            idx.k_norm_gamma,
                            idx_kv_pool,
                            bt_idx_kv,
                            idx_st_pool,
                            bt_idx_st,
                            kv_tokens_per_block=idx_kv_tpb,
                            state_tokens_per_block=idx_st_tpb,
                            ratio=idx.compress_ratio,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            norm_eps=idx.norm_eps,
                            valid=active_rows,
                        )
                        max_ctx_l = max(1, (max_prev + 1) // idx.compress_ratio)
                        src_selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            idx_kv_pool,
                            bt_idx_kv,
                            vis_t,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            max_ctx_len=max_ctx_l,
                            **cp_idx_kw,
                        )
                        source_idx_local[lid] = (idx_kv_pool, bt_idx_kv)

                    # NOTE: FullAttn MQA-1 needs exact per-request valid KV slice when batch has
                    # different sequence_lengths! When all seq_lens_before match, batched attention is exact.
                    if len(set(seq_lens_before)) == 1:
                        attn_out = sa(
                            attn_in,
                            positions=positions_2d,
                            kv_cache=ephemeral_kv,
                            k_sconv_state=k_sc_st,
                            v_sconv_state=v_sc_st,
                            seq_offset=max_prev,
                            qsa_selected=src_selected,
                            preprojected=preproj,
                        )
                    else:
                        req_outs: List[torch.Tensor] = []
                        for b, plen in enumerate(seq_lens_before):
                            k_b = ephemeral_kv.k_cache[b : b + 1, : plen + 1].clone()
                            v_b = ephemeral_kv.v_cache[b : b + 1, : plen + 1].clone()
                            kv_b = FullAttnKVCache(k_cache=k_b, v_cache=v_b, seq_len=plen)
                            ks_b = ShortConvState(k_sc_st.conv_state[b : b + 1]) if k_sc_st is not None else None
                            vs_b = ShortConvState(v_sc_st.conv_state[b : b + 1]) if v_sc_st is not None else None
                            sel_b = src_selected[b : b + 1] if src_selected is not None else None
                            pre_b = tuple(t[b : b + 1] if t is not None else None for t in preproj)
                            out_b = sa(
                                attn_in[b : b + 1],
                                positions=positions_2d[b : b + 1],
                                kv_cache=kv_b,
                                k_sconv_state=ks_b,
                                v_sconv_state=vs_b,
                                seq_offset=plen,
                                qsa_selected=sel_b,
                                preprojected=pre_b,
                            )
                            ephemeral_kv.k_cache[b, plen].copy_(kv_b.k_cache[0, plen])
                            ephemeral_kv.v_cache[b, plen].copy_(kv_b.v_cache[0, plen])
                            req_outs.append(out_b)
                        ephemeral_kv.seq_len = max_prev + 1
                        attn_out = torch.cat(req_outs, dim=0)

                    # Write single new K/V slot per request to physical KV pool
                    tpb = int(kv_c.seq_size_per_block)
                    for b, plen in enumerate(seq_lens_before):
                        k_tok = ephemeral_kv.k_cache[b : b + 1, plen] if len(set(seq_lens_before)) != 1 else ephemeral_kv.k_cache[b : b + 1, max_prev]
                        v_tok = ephemeral_kv.v_cache[b : b + 1, plen] if len(set(seq_lens_before)) != 1 else ephemeral_kv.v_cache[b : b + 1, max_prev]
                        _store_single_req_full_attn_kv(
                            kv_c.kv_cache_base,
                            bt_kv[b],
                            k_tok,
                            v_tok,
                            plen,
                            tpb,
                            num_kv_heads=self.cfg.num_key_value_heads,
                            head_dim=self.cfg.head_dim,
                            allow_zero_padded=is_padded,
                            kv_scale=getattr(kv_c, "kv_scale_base", None),
                        )
                    if len(set(seq_lens_before)) == 1:
                        # Repack contiguous valid KV for consumer layers on this step
                        full_kv_ephemeral[lid] = ephemeral_kv
                    else:
                        # Build per-request slice map for consumer layers
                        full_kv_ephemeral[lid] = FullAttnKVCache(
                            k_cache=torch.empty(0), v_cache=torch.empty(0), seq_len=max_prev + 1
                        )
                        full_kv_ephemeral[lid]._per_req = (kv_c.kv_cache_base, bt_kv, tpb)  # type: ignore[attr-defined]

                    if sc_c is not None and bt_sc is not None and k_sc_st is not None and v_sc_st is not None:
                        packed_sc = torch.stack([k_sc_st.conv_state, v_sc_st.conv_state], dim=1)
                        self._write_opaque_batch(
                            sc_c,
                            bt_sc,
                            new_lens,
                            packed_sc,
                            FULL_ATTN_KV_SCONV_TAG,
                            lid,
                            allow_zero_padded=is_padded,
                        )
                else:
                    src_lid = layer.kv_cache_owner
                    cons_selected: Optional[torch.Tensor] = None
                    if sa.indexer is not None:
                        idx = sa.indexer
                        src_idx_kv_pool, src_bt_idx_kv = source_idx_local[src_lid]
                        assert preproj[3] is not None
                        q_idx = preproj[3]
                        seq_b_t = torch.tensor(
                            seq_lens_before, dtype=torch.int32, device=device
                        )
                        active_rows = (
                            (src_bt_idx_kv != 0).any(dim=1) if is_padded else None
                        )
                        vis_t = (
                            torch.where(active_rows, seq_b_t + 1, 0)
                            if active_rows is not None
                            else (seq_b_t + 1)
                        )
                        max_ctx_l = max(1, (max_prev + 1) // idx.compress_ratio)
                        cons_selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            src_idx_kv_pool,
                            src_bt_idx_kv,
                            vis_t,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=layer.self_attn.indexer_rotary_dim,
                            rope_theta=layer.self_attn.rope_theta,
                            max_ctx_len=max_ctx_l,
                            **cp_idx_kw,
                        )
                    shared_kv = full_kv_ephemeral[src_lid]
                    per_req = getattr(shared_kv, "_per_req", None)
                    if per_req is None:
                        attn_out = sa(
                            attn_in,
                            positions=positions_2d,
                            kv_cache=shared_kv,
                            seq_offset=max_prev,
                            qsa_selected=cons_selected,
                            preprojected=preproj,
                        )
                    else:
                        src_base, src_bt, src_tpb = per_req
                        req_outs = []
                        for b, plen in enumerate(seq_lens_before):
                            k_req, v_req = _load_single_req_full_attn_kv(
                                src_base,
                                src_bt[b],
                                plen + 1,
                                src_tpb,
                                num_kv_heads=self.cfg.num_key_value_heads,
                                head_dim=self.cfg.head_dim,
                                allow_zero_padded=is_padded,
                            )
                            kv_b = FullAttnKVCache(
                                k_cache=k_req.unsqueeze(0),
                                v_cache=v_req.unsqueeze(0),
                                seq_len=plen + 1,
                            )
                            sel_b = cons_selected[b : b + 1] if cons_selected is not None else None
                            pre_b = tuple(t[b : b + 1] if t is not None else None for t in preproj)
                            req_outs.append(
                                sa(
                                    attn_in[b : b + 1],
                                    positions=positions_2d[b : b + 1],
                                    kv_cache=kv_b,
                                    seq_offset=plen,
                                    qsa_selected=sel_b,
                                    preprojected=pre_b,
                                )
                            )
                        attn_out = torch.cat(req_outs, dim=0)

            # 4. Attention Output ShortConv
            if layer.attn_output_sconv is not None:
                tag_out = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao = layer_caches[tag_out]
                bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                ao_state = ShortConvState(
                    self._read_opaque_batch(
                        c_ao,
                        bt_ao,
                        seq_lens_before,
                        tag_out,
                        lid,
                        target_dtype=dtype,
                        allow_zero_padded=is_padded,
                    )
                )
                attn_out = layer.attn_output_sconv(attn_out, state=ao_state)
                self._write_opaque_batch(
                    c_ao,
                    bt_ao,
                    new_lens,
                    ao_state.conv_state,
                    tag_out,
                    lid,
                    allow_zero_padded=is_padded,
                )
            if self.cfg.clip_attn_output is not None:
                attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)

            # 5. MLP Hyper-Connection + MLP
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                layer.mlp_hyper_connection.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
            assert mlp_inj is not None
            mlp_out = _run_runtime_mlp(layer, mlp_in)

            # 6. FFN Output ShortConv + clamp
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                fo_state = ShortConvState(
                    self._read_opaque_batch(
                        c_fo,
                        bt_fo,
                        seq_lens_before,
                        FFN_OUTPUT_SCONV_TAG,
                        lid,
                        target_dtype=dtype,
                        allow_zero_padded=is_padded,
                    )
                )
                mlp_out = layer.ffn_output_sconv(mlp_out, state=fo_state)
                self._write_opaque_batch(
                    c_fo,
                    bt_fo,
                    new_lens,
                    fo_state.conv_state,
                    FFN_OUTPUT_SCONV_TAG,
                    lid,
                    allow_zero_padded=is_padded,
                )
            if self.cfg.clip_mlp_output is not None:
                mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)

            deferred = (mlp_out, mlp_inj)

        assert deferred is not None
        hidden, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
            deferred[0], hyper_hidden, deferred[1], prev_latent=None
        )
        hidden_out = hidden.reshape(bsz, self.cfg.hidden_size)
        if aux_out is not None:
            return hidden_out, aux_out
        return hidden_out

    def run_prefill_ragged(
        self,
        model: Qwen4_5Model,
        flat_input_ids: torch.Tensor,
        flat_positions: Optional[torch.Tensor],
        input_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
        kv_cache: Any,
        attention_inputs: Any,
        writers: Mapping[str, Any],
        *,
        prefill_handle: Optional[Qwen4_5PrefillHandle] = None,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Whole-batch packed ragged prefill preserving exact block-boundary snapshots."""
        with _model_a2a_step_scope(model):
            return self._run_prefill_ragged_impl(
                model,
                flat_input_ids,
                flat_positions,
                input_lengths,
                prefix_lengths,
                kv_cache,
                attention_inputs,
                writers,
                prefill_handle=prefill_handle,
                enable_yoco_decode=enable_yoco_decode,
                capture_layer_ids=capture_layer_ids,
            )

    def _run_prefill_ragged_impl(
        self,
        model: Qwen4_5Model,
        flat_input_ids: torch.Tensor,
        flat_positions: Optional[torch.Tensor],
        input_lengths: Sequence[int],
        prefix_lengths: Sequence[int],
        kv_cache: Any,
        attention_inputs: Any,
        writers: Mapping[str, Any],
        *,
        prefill_handle: Optional[Qwen4_5PrefillHandle] = None,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = flat_input_ids.device
        dtype = model.embed_tokens.weight.dtype
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )
        is_cg = bool(prefill_handle is not None and prefill_handle.is_cuda_graph)
        plan = (
            prefill_handle.plan
            if prefill_handle is not None
            else self._build_ragged_prefill_plan(input_lengths, prefix_lengths, device)
        )
        total_tokens = int(flat_input_ids.numel())
        if flat_positions is None:
            flat_positions = (
                prefill_handle.flat_pos
                if prefill_handle is not None and prefill_handle.flat_pos is not None
                else plan.tok_abs_pos_i64
            )
        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None

        if plan.bsz == 0 or total_tokens == 0:
            self._run_empty_ep_collectives(model, dtype, device)
            empty_out = torch.empty((0, self.cfg.hidden_size), dtype=dtype, device=device)
            if capture_layer_ids:
                aux_out = torch.empty(
                    (0, len(capture_layer_ids) * self.cfg.hidden_size),
                    dtype=dtype,
                    device=device,
                )
                return empty_out, aux_out
            return empty_out

        eff_sp, eff_rk, eff_grp, eff_comm = self._resolve_sp_dist()
        sp_active = eff_sp > 1 and total_tokens > 0
        if sp_active:
            sub_plan = self._build_ragged_sp_subplan(plan, eff_sp, eff_rk, device)
            s0, s1 = sub_plan.s0, sub_plan.s1
            n_loc = sub_plan.n_loc
        else:
            sub_plan = None
            s0, s1, n_loc = 0, total_tokens, total_tokens

        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = torch.empty(
                (n_loc, len(capture_layer_ids) * self.cfg.hidden_size),
                dtype=dtype,
                device=device,
            )

        def _active_bt(tag: str) -> torch.Tensor:
            bt = _block_table_for_tag(attention_inputs, tag)
            return bt if plan.act_idx_dev is None else bt.index_select(0, plan.act_idx_dev)

        _snap_phys_cache: Dict[
            Tuple[int, int, int, int, int],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool],
        ] = {}

        def _snap_phys_for(cache: Any, bt_act: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool
        ]:
            tpb_c = int(cache.seq_size_per_block)
            max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
            s0 = int(bt_act.stride(0))
            s1 = int(bt_act.stride(1)) if bt_act.ndim >= 2 else 0
            ck = (tpb_c, bt_act.data_ptr(), max_cols, s0, s1)
            hit = _snap_phys_cache.get(ck)
            if hit is not None:
                return hit
            res_s = self._compute_snap_phys(plan, tpb_c, bt_act, prefill_handle, device)
            _snap_phys_cache[ck] = res_s
            return res_s

        _sp_snap_phys_cache: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}

        def _sp_snap_phys_for(cache: Any, bt_act: torch.Tensor) -> torch.Tensor:
            tpb_c = int(cache.seq_size_per_block)
            max_cols = int(bt_act.shape[1]) if bt_act.ndim >= 2 else 0
            bs0 = int(bt_act.stride(0))
            bs1 = int(bt_act.stride(1)) if bt_act.ndim >= 2 else 0
            ck = (tpb_c, bt_act.data_ptr(), max_cols, bs0, bs1)
            hit_sp = _sp_snap_phys_cache.get(ck)
            if hit_sp is not None:
                return hit_sp
            _, _, _, _, s_phys_all, _, _ = _snap_phys_for(cache, bt_act)
            assert sub_plan is not None
            s_phys_sub = self._compute_sp_snap_phys(plan, sub_plan, tpb_c, bt_act, s_phys_all, prefill_handle, device)
            _sp_snap_phys_cache[ck] = s_phys_sub
            return s_phys_sub

        use_static_init = is_cg or plan.bsz <= 16
        init_scratch_store = prefill_handle.init_bufs if prefill_handle is not None else self._prefill_init_scratch
        dev_idx_k = device.index if device.index is not None else -1

        def _read_init_state(
            cache: Any,
            bt_act: torch.Tensor,
            tag: str,
            lid: int,
            shape_tail: Tuple[int, ...],
            target_dtype: torch.dtype,
            fill_value: int = 0,
        ) -> Optional[torch.Tensor]:
            if not plan.has_any_prefix:
                return None
            if device.type == "cuda":
                if (use_static_init and not torch.cuda.is_current_stream_capturing()) or (prefill_handle is not None):
                    sk = (dev_idx_k, plan.bsz, tag, lid, shape_tail, target_dtype)
                    s_buf = init_scratch_store.get(sk)
                    if s_buf is None or s_buf.device != device:
                        s_buf = torch.empty((plan.bsz, *shape_tail), dtype=target_dtype, device=device)
                        init_scratch_store[sk] = s_buf
                    return self._read_opaque_batch_static_gpu(
                        cache,
                        bt_act,
                        plan.prefix_i32,
                        s_buf,
                        tag,
                        lid,
                        fill_value=fill_value,
                    )
                return self._read_opaque_batch_gpu(
                    cache,
                    bt_act,
                    plan.prefix_i32,
                    tag,
                    lid,
                    fill_value=fill_value,
                    target_dtype=target_dtype,
                )
            return self._read_opaque_batch(
                cache,
                bt_act,
                plan.active_pfx,
                tag,
                lid,
                fill_value=fill_value,
                target_dtype=target_dtype,
            )

        emb_ids = flat_input_ids[s0:s1] if sp_active else flat_input_ids
        emb_flat = model.embed_tokens(emb_ids)
        hyper_hidden = (
            emb_flat
            if self.cfg.hc_count > 1 and (len(model.layers) == 0 or model.layers[0].ple is None)
            else emb_flat.repeat(1, self.cfg.hc_count)
        )
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        shared_gdn_packed: Dict[int, SharedGDNInputs] = {}
        source_prefix_ssm: Dict[int, Optional[torch.Tensor]] = {}
        full_kv_reqs: Dict[int, List[FullAttnKVCache]] = {}
        full_kv_batched: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        full_qsa_raw_reqs: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        full_qsa_raw_batched: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        source_idx_prefill: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        source_kv_prefill: Dict[int, Tuple[Any, torch.Tensor]] = {}
        sp_snap_sync_specs: List[Tuple[Any, int, torch.Tensor, int, torch.Tensor]] = []
        vis_q_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        sdpa_mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        def _get_vis_q(b_v: int, t_pfx_v: int, t_in_v: int) -> torch.Tensor:
            if (
                prefill_handle is not None
                and prefill_handle.static_vis_q is not None
                and tuple(prefill_handle.static_vis_q.shape) == (b_v, t_in_v)
            ):
                return prefill_handle.static_vis_q
            vk = (b_v, t_pfx_v, t_in_v)
            hit = vis_q_cache.get(vk)
            if hit is None:
                hit = (
                    torch.arange(t_pfx_v + 1, t_pfx_v + t_in_v + 1, dtype=torch.int32, device=device)
                    .unsqueeze(0)
                    .expand(b_v, t_in_v)
                )
                vis_q_cache[vk] = hit
            return hit

        def _get_sdpa_mask(t_pfx_v: int, t_tot_v: int) -> torch.Tensor:
            mk = (t_pfx_v, t_tot_v)
            hit = sdpa_mask_cache.get(mk)
            if hit is None:
                q_i = torch.arange(t_pfx_v, t_tot_v, device=device).unsqueeze(1)
                k_i = torch.arange(t_tot_v, device=device).unsqueeze(0)
                hit = q_i >= k_i
                sdpa_mask_cache[mk] = hit
            return hit

        cp_idx_kw = self._cp_indexer_kwargs()
        uniform_full_batch = plan.is_uniform_batch

        def _ensure_source_kv_batched(src_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if src_id not in full_kv_batched:
                src_kv_c, src_bt_act = source_kv_prefill[src_id]
                bsz_u = plan.bsz
                t_tot_u = plan.active_pfx[0] + plan.active_lens[0]
                k_fb, v_fb = _load_batched_full_attn_kv(
                    src_kv_c.kv_cache_base,
                    src_bt_act,
                    bsz_u,
                    t_tot_u,
                    int(src_kv_c.seq_size_per_block),
                    num_kv_heads=sa.num_kv_heads,
                    head_dim=sa.head_dim,
                    kv_scale=getattr(src_kv_c, "kv_scale_base", None),
                    out_dtype=dtype,
                )
                full_kv_batched[src_id] = (k_fb, v_fb)
            return full_kv_batched[src_id]

        flat_pos_2d = flat_positions.unsqueeze(0)

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = (
                {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}
                if kv_cache is not None
                else {}
            )

            # 1. Optional PLE (Layer 1) — packed hashing, 1 gather + 1 AllReduce, 1 W_kv GEMM, 1 conv+scatter + fused HC combine/norm
            ple_hc_normed: Optional[torch.Tensor] = None
            if layer.ple is not None:
                pre_sub = deferred[0] if deferred is not None else None
                pre_inj = deferred[1] if deferred is not None else None
                deferred = None
                attn_hc = layer.attn_hyper_connection

                if kv_cache is not None:
                    c_st = layer_caches[PLE_STATE_TAG]
                    c_ctx = layer_caches[PLE_NGRAM_CTX_TAG]
                    bt_st = _active_bt(PLE_STATE_TAG)
                    bt_ctx = _active_bt(PLE_NGRAM_CTX_TAG)
                    init_ctx = _read_init_state(
                        c_ctx,
                        bt_ctx,
                        PLE_NGRAM_CTX_TAG,
                        lid,
                        (self.cfg.ngram_size - 1,),
                        torch.int64,
                        fill_value=self.cfg.eos_token_id,
                    )
                    init_conv = _read_init_state(
                        c_st,
                        bt_st,
                        PLE_STATE_TAG,
                        lid,
                        (self.cfg.hc_count * self.cfg.hidden_size, max(0, self.cfg.ple_conv_kernel_size - 1)),
                        dtype,
                    )
                    conv_pool_bf16, _ = _view_opaque_pool_2d(c_st.kv_cache_base, dtype)
                    ctx_pool_i64, _ = _view_opaque_pool_2d(c_ctx.kv_cache_base, torch.int64)
                    if sp_active:
                        assert sub_plan is not None
                        _, _, _, _, s_phys_st_all, _, _ = _snap_phys_for(c_st, bt_st)
                        _, _, _, _, s_phys_ctx_all, _, _ = _snap_phys_for(c_ctx, bt_ctx)
                        sp_snap_sync_specs.append(
                            (c_st.kv_cache_base, int(c_st.seq_size_per_block), bt_st, self.cfg.hc_count * self.cfg.hidden_size * max(0, self.cfg.ple_conv_kernel_size - 1) * 2, s_phys_st_all)
                        )
                        sp_snap_sync_specs.append(
                            (c_ctx.kv_cache_base, int(c_ctx.seq_size_per_block), bt_ctx, max(0, self.cfg.ngram_size - 1) * 8, s_phys_ctx_all)
                        )
                        if sub_plan.sub_bsz > 0:
                            if init_ctx is not None:
                                sub_init_ctx = init_ctx.index_select(0, sub_plan.sub_req_idx_dev)
                            else:
                                sub_init_ctx = torch.full(
                                    (sub_plan.sub_bsz, max(0, self.cfg.ngram_size - 1)),
                                    self.cfg.eos_token_id,
                                    dtype=torch.int64,
                                    device=device,
                                )
                            if sub_plan.loc_start == 1:
                                sub_init_ctx[0, 0] = sub_init_ctx[0, 1]
                                sub_init_ctx[0, 1] = flat_input_ids[s0 - 1].to(torch.int64)
                            elif sub_plan.loc_start >= 2:
                                sub_init_ctx[0, 0] = flat_input_ids[s0 - 2].to(torch.int64)
                                sub_init_ctx[0, 1] = flat_input_ids[s0 - 1].to(torch.int64)
                            sub_init_conv = init_conv.index_select(0, sub_plan.sub_req_idx_dev) if init_conv is not None else None
                            if sub_init_conv is not None and sub_plan.loc_start > 0:
                                pw = max(0, self.cfg.ple_conv_kernel_size - 1)
                                if sub_plan.loc_start >= pw:
                                    sub_init_conv[0].zero_()
                                else:
                                    keep_p = pw - sub_plan.loc_start
                                    sub_init_conv[0, :, :keep_p] = init_conv[sub_plan.sub_reqs[0], :, sub_plan.loc_start :]
                                    sub_init_conv[0, :, keep_p:].zero_()
                        else:
                            sub_init_ctx = init_ctx
                            sub_init_conv = init_conv
                        s_phys_st_ov = _sp_snap_phys_for(c_st, bt_st)
                        s_phys_ctx_ov = _sp_snap_phys_for(c_ctx, bt_ctx)
                        s_tok_st, s_req_st, s_end_st, r_ss_st, s_phys_st, a16_st, _ = self._get_sp_snap_schedule(
                            plan, sub_plan, int(c_st.seq_size_per_block), bt_st, device, s_phys_override=s_phys_st_ov
                        )
                        _, _, _, _, s_phys_ctx, _, _ = self._get_sp_snap_schedule(
                            plan, sub_plan, int(c_ctx.seq_size_per_block), bt_ctx, device, s_phys_override=s_phys_ctx_ov
                        )
                        hyper_hidden, ple_hc_normed = layer.ple.forward_packed_prefill_with_hc(
                            hyper_hidden,
                            emb_ids,
                            sub_plan.cu_seqlens_i32,
                            sub_plan.tok_req_i32,
                            sub_plan.tile_req_16,
                            sub_plan.tile_t0_16,
                            prev_ngram_ctx=sub_init_ctx,
                            init_conv_state=sub_init_conv,
                            snap_tok_idx=s_tok_st,
                            snap_req_idx=s_req_st,
                            conv_dst_pool=conv_pool_bf16,
                            conv_snap_phys=s_phys_st,
                            req_snap_start=r_ss_st,
                            snap_local_ends=s_end_st,
                            snap_aligned=a16_st,
                            ctx_dst_pool=ctx_pool_i64,
                            ctx_snap_phys=s_phys_ctx,
                            skip_invalid_phys=True,
                            block_t=16,
                            pre_combine_sub=pre_sub,
                            pre_combine_inj=pre_inj,
                            hc_norm_weight=attn_hc.hc_norm_weight,
                            hc_norm_bias=attn_hc.hc_norm_bias,
                            hc_eps=attn_hc.eps,
                            sp_tokens_sharded=True,
                            sp_total_tokens=total_tokens,
                            sp_halo_exchange=sub_plan.has_straddle,
                            sp_halo_loc_start=sub_plan.loc_start,
                            sp_halo_req0_start=sub_plan.req0_start,
                            sp_size=eff_sp,
                            sp_rank=eff_rk,
                            sp_group=eff_grp,
                            sp_fast_comm=eff_comm,
                        )
                    else:
                        s_tok_st, s_req_st, s_end_st, r_ss_st, s_phys_st, a16_st, _ = _snap_phys_for(c_st, bt_st)
                        _, _, _, _, s_phys_ctx, _, _ = _snap_phys_for(c_ctx, bt_ctx)
                        hyper_hidden, ple_hc_normed = layer.ple.forward_packed_prefill_with_hc(
                            hyper_hidden,
                            flat_input_ids,
                            plan.cu_seqlens_i32,
                            plan.tok_req_i32,
                            plan.tile_req_16,
                            plan.tile_t0_16,
                            prev_ngram_ctx=init_ctx,
                            init_conv_state=init_conv,
                            snap_tok_idx=s_tok_st,
                            snap_req_idx=s_req_st,
                            conv_dst_pool=conv_pool_bf16,
                            conv_snap_phys=s_phys_st,
                            req_snap_start=r_ss_st,
                            snap_local_ends=s_end_st,
                            snap_aligned=a16_st,
                            ctx_dst_pool=ctx_pool_i64,
                            ctx_snap_phys=s_phys_ctx,
                            skip_invalid_phys=True,
                            block_t=16,
                            pre_combine_sub=pre_sub,
                            pre_combine_inj=pre_inj,
                            hc_norm_weight=attn_hc.hc_norm_weight,
                            hc_norm_bias=attn_hc.hc_norm_bias,
                            hc_eps=attn_hc.eps,
                        )
                else:
                    if sp_active:
                        assert sub_plan is not None
                        sub_init_ctx_n = None
                        if sub_plan.sub_bsz > 0 and sub_plan.loc_start > 0:
                            sub_init_ctx_n = torch.full(
                                (sub_plan.sub_bsz, max(0, self.cfg.ngram_size - 1)),
                                self.cfg.eos_token_id,
                                dtype=torch.int64,
                                device=device,
                            )
                            if sub_plan.loc_start == 1:
                                sub_init_ctx_n[0, 1] = flat_input_ids[s0 - 1].to(torch.int64)
                            else:
                                sub_init_ctx_n[0, 0] = flat_input_ids[s0 - 2].to(torch.int64)
                                sub_init_ctx_n[0, 1] = flat_input_ids[s0 - 1].to(torch.int64)
                        hyper_hidden, ple_hc_normed = layer.ple.forward_packed_prefill_with_hc(
                            hyper_hidden,
                            emb_ids,
                            sub_plan.cu_seqlens_i32,
                            sub_plan.tok_req_i32,
                            sub_plan.tile_req_16,
                            sub_plan.tile_t0_16,
                            prev_ngram_ctx=sub_init_ctx_n,
                            block_t=16,
                            pre_combine_sub=pre_sub,
                            pre_combine_inj=pre_inj,
                            hc_norm_weight=attn_hc.hc_norm_weight,
                            hc_norm_bias=attn_hc.hc_norm_bias,
                            hc_eps=attn_hc.eps,
                            sp_tokens_sharded=True,
                            sp_total_tokens=total_tokens,
                            sp_halo_exchange=sub_plan.has_straddle,
                            sp_halo_loc_start=sub_plan.loc_start,
                            sp_halo_req0_start=sub_plan.req0_start,
                            sp_size=eff_sp,
                            sp_rank=eff_rk,
                            sp_group=eff_grp,
                            sp_fast_comm=eff_comm,
                        )
                    else:
                        hyper_hidden, ple_hc_normed = layer.ple.forward_packed_prefill_with_hc(
                            hyper_hidden,
                            flat_input_ids,
                            plan.cu_seqlens_i32,
                            plan.tok_req_i32,
                            plan.tile_req_16,
                            plan.tile_t0_16,
                            block_t=16,
                            pre_combine_sub=pre_sub,
                            pre_combine_inj=pre_inj,
                            hc_norm_weight=attn_hc.hc_norm_weight,
                            hc_norm_bias=attn_hc.hc_norm_bias,
                            hc_eps=attn_hc.eps,
                        )

            # 2. Attn Hyper-Connection (1 pass across ALL N_loc tokens)
            if ple_hc_normed is not None:
                attn_in, hyper_hidden, attn_inj, prev_latent = self._hc_mix_from_normed_safe(
                    layer.attn_hyper_connection, ple_hc_normed, hyper_hidden, prev_latent
                )
            elif deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = self._hc_mix_with_latent_safe(
                    layer.attn_hyper_connection, hyper_hidden, prev_latent
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = self._hc_combine_and_mix_safe(
                    layer.attn_hyper_connection, deferred[0], hyper_hidden, deferred[1], prev_latent
                )
                deferred = None
            assert attn_inj is not None

            if aux_out is not None and lid in cap_slots:
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * self.cfg.hidden_size : (slot_i + 1) * self.cfg.hidden_size].copy_(
                        attn_in
                    )

            if sp_active:
                attn_in = self._sp_all_gather_dim0(
                    attn_in,
                    total_tokens=total_tokens,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                )

            # 3. Attention Core (Packed GDN or Whole-Sequence QSA / FullAttn)
            if layer.linear_attn is not None:
                if layer.is_consumer:
                    init_q_conv: Optional[torch.Tensor] = None
                    init_c_ssm: Optional[torch.Tensor] = None
                    c_snap_tok = c_snap_req = c_snap_phys = c_pool_bf16 = None
                    c_dst_col_off = 0
                    s_req_start = s_loc_end = s_phys_ssm = s_pool_fp32 = None
                    c_direct_64 = False

                    if kv_cache is not None and LINEAR_ATTN_TAG in layer_caches:
                        lin_c = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin_c = _active_bt(LINEAR_ATTN_TAG)
                        init_packed = _read_init_state(
                            lin_c,
                            bt_lin_c,
                            LINEAR_ATTN_TAG,
                            lid,
                            (int(lin_c.kv_cache_base.shape[1]),),
                            torch.float32,
                        )
                        ssm_n = math.prod(self.ssm_shape)
                        conv_n_c = math.prod(self.gdn_consumer_conv_shape)
                        if init_packed is not None:
                            init_c_ssm = init_packed[:, :ssm_n].view(plan.bsz, *self.ssm_shape)
                            init_q_conv = (
                                init_packed[:, ssm_n:]
                                .view(torch.bfloat16)[:, :conv_n_c]
                                .view(plan.bsz, *self.gdn_consumer_conv_shape)
                                .to(dtype)
                            )
                        s_tok, s_req, s_end, r_sstart, s_phys, _, _ = _snap_phys_for(lin_c, bt_lin_c)
                        c_snap_tok, c_snap_req, c_snap_phys = s_tok, s_req, s_phys
                        c_pool_bf16, _ = _view_opaque_pool_2d(lin_c.kv_cache_base, dtype)
                        c_dst_col_off = 2 * ssm_n
                        s_req_start, s_loc_end, s_phys_ssm = r_sstart, s_end, s_phys
                        s_pool_fp32, _ = _view_opaque_pool_2d(lin_c.kv_cache_base, torch.float32)
                        c_direct_64 = (
                            int(lin_c.seq_size_per_block) == 64
                            and all((p & 63) == 0 for p in plan.active_pfx)
                        )
                    elif kv_cache is not None and GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _active_bt(GDN_SSM_STATE_TAG)
                        init_c_ssm = _read_init_state(
                            c_ssm, bt_ssm, GDN_SSM_STATE_TAG, lid, self.ssm_shape, torch.float32
                        )
                        _, _, s_end, r_sstart, s_phys, _, _ = _snap_phys_for(c_ssm, bt_ssm)
                        s_req_start, s_loc_end, s_phys_ssm = r_sstart, s_end, s_phys
                        s_pool_fp32, _ = _view_opaque_pool_2d(c_ssm.kv_cache_base, torch.float32)
                        c_direct_64 = (
                            int(c_ssm.seq_size_per_block) == 64
                            and all((p & 63) == 0 for p in plan.active_pfx)
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_q = layer_caches[GDN_CONV_STATE_TAG]
                            bt_q = _active_bt(GDN_CONV_STATE_TAG)
                            init_q_conv = _read_init_state(
                                c_q, bt_q, GDN_CONV_STATE_TAG, lid, self.gdn_consumer_conv_shape, dtype
                            )
                            s_tok, s_req, _, _, q_phys, _, _ = _snap_phys_for(c_q, bt_q)
                            c_snap_tok, c_snap_req, c_snap_phys = s_tok, s_req, q_phys
                            c_pool_bf16, _ = _view_opaque_pool_2d(c_q.kv_cache_base, dtype)
                    elif kv_cache is not None and GDN_CONSUMER_Q_CONV_TAG in layer_caches:
                        c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                        bt_q = _active_bt(GDN_CONSUMER_Q_CONV_TAG)
                        init_q_conv = _read_init_state(
                            c_q, bt_q, GDN_CONSUMER_Q_CONV_TAG, lid, self.gdn_consumer_conv_shape, dtype
                        )
                        if yoco_on and plan.has_any_prefix and layer.source_layer_idx is not None:
                            init_c_ssm = source_prefix_ssm.get(layer.source_layer_idx)
                        s_tok, s_req, _, _, q_phys, _, _ = _snap_phys_for(c_q, bt_q)
                        c_snap_tok, c_snap_req, c_snap_phys = s_tok, s_req, q_phys
                        c_pool_bf16, _ = _view_opaque_pool_2d(c_q.kv_cache_base, dtype)

                    attn_out, _ = layer.linear_attn.forward_packed_prefill(
                        attn_in,
                        plan.cu_seqlens_i32,
                        plan.tile_req_32,
                        plan.tile_t0_32,
                        plan.chunk_req_64,
                        plan.chunk_t0_64,
                        plan.req_chunk_start_64,
                        init_conv_state=init_q_conv,
                        init_ssm_state=init_c_ssm,
                        shared=shared_gdn_packed[layer.source_layer_idx],  # type: ignore[index]
                        snap_tok_idx=c_snap_tok,
                        snap_req_idx=c_snap_req,
                        conv_dst_pool=c_pool_bf16,
                        conv_snap_phys=c_snap_phys,
                        conv_dst_col_offset=c_dst_col_off,
                        req_snap_start=s_req_start,
                        snap_local_end=s_loc_end,
                        ssm_dst_pool=s_pool_fp32,
                        ssm_snap_phys=s_phys_ssm,
                        skip_invalid_phys=True,
                        snaps_direct_chunk64=c_direct_64,
                        block_t=32,
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=sp_active,
                    )
                else:
                    init_conv: Optional[torch.Tensor] = None
                    init_ssm: Optional[torch.Tensor] = None
                    c_snap_tok = c_snap_req = c_snap_phys = c_pool_bf16 = None
                    c_dst_col_off = 0
                    s_req_start = s_loc_end = s_phys_ssm = s_pool_fp32 = None
                    s_direct_64 = False

                    if kv_cache is not None and GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _active_bt(GDN_SSM_STATE_TAG)
                        init_ssm = _read_init_state(
                            c_ssm, bt_ssm, GDN_SSM_STATE_TAG, lid, self.ssm_shape, torch.float32
                        )
                        _, _, s_end, r_sstart, s_phys, _, _ = _snap_phys_for(c_ssm, bt_ssm)
                        s_req_start, s_loc_end, s_phys_ssm = r_sstart, s_end, s_phys
                        s_pool_fp32, _ = _view_opaque_pool_2d(c_ssm.kv_cache_base, torch.float32)
                        s_direct_64 = (
                            int(c_ssm.seq_size_per_block) == 64
                            and all((p & 63) == 0 for p in plan.active_pfx)
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_conv = _active_bt(GDN_CONV_STATE_TAG)
                            init_conv = _read_init_state(
                                c_conv, bt_conv, GDN_CONV_STATE_TAG, lid, self.gdn_conv_shape, dtype
                            )
                            s_tok, s_req, _, _, conv_phys, _, _ = _snap_phys_for(c_conv, bt_conv)
                            c_snap_tok, c_snap_req, c_snap_phys = s_tok, s_req, conv_phys
                            c_pool_bf16, _ = _view_opaque_pool_2d(c_conv.kv_cache_base, dtype)
                    elif kv_cache is not None and LINEAR_ATTN_TAG in layer_caches:
                        lin_c = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin = _active_bt(LINEAR_ATTN_TAG)
                        init_packed = _read_init_state(
                            lin_c,
                            bt_lin,
                            LINEAR_ATTN_TAG,
                            lid,
                            (int(lin_c.kv_cache_base.shape[1]),),
                            torch.float32,
                        )
                        ssm_n = math.prod(self.ssm_shape)
                        conv_n = math.prod(self.gdn_conv_shape)
                        if init_packed is not None:
                            init_ssm = init_packed[:, :ssm_n].view(plan.bsz, *self.ssm_shape)
                            init_conv = (
                                init_packed[:, ssm_n:]
                                .view(torch.bfloat16)[:, :conv_n]
                                .view(plan.bsz, *self.gdn_conv_shape)
                                .to(dtype)
                            )
                        s_tok, s_req, s_end, r_sstart, s_phys, _, _ = _snap_phys_for(lin_c, bt_lin)
                        c_snap_tok, c_snap_req, c_snap_phys = s_tok, s_req, s_phys
                        c_pool_bf16, _ = _view_opaque_pool_2d(lin_c.kv_cache_base, dtype)
                        c_dst_col_off = 2 * ssm_n
                        s_req_start, s_loc_end, s_phys_ssm = r_sstart, s_end, s_phys
                        s_pool_fp32, _ = _view_opaque_pool_2d(lin_c.kv_cache_base, torch.float32)
                        s_direct_64 = (
                            int(lin_c.seq_size_per_block) == 64
                            and all((p & 63) == 0 for p in plan.active_pfx)
                        )

                    if layer.is_source:
                        source_prefix_ssm[lid] = init_ssm.clone() if init_ssm is not None else None

                    attn_out, published = layer.linear_attn.forward_packed_prefill(
                        attn_in,
                        plan.cu_seqlens_i32,
                        plan.tile_req_32,
                        plan.tile_t0_32,
                        plan.chunk_req_64,
                        plan.chunk_t0_64,
                        plan.req_chunk_start_64,
                        init_conv_state=init_conv,
                        init_ssm_state=init_ssm,
                        shared=None,
                        snap_tok_idx=c_snap_tok,
                        snap_req_idx=c_snap_req,
                        conv_dst_pool=c_pool_bf16,
                        conv_snap_phys=c_snap_phys,
                        conv_dst_col_offset=c_dst_col_off,
                        req_snap_start=s_req_start,
                        snap_local_end=s_loc_end,
                        ssm_dst_pool=s_pool_fp32,
                        ssm_snap_phys=s_phys_ssm,
                        skip_invalid_phys=True,
                        snaps_direct_chunk64=s_direct_64,
                        block_t=32,
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=sp_active,
                    )
                    if published is not None:
                        shared_gdn_packed[lid] = published
            else:
                assert layer.self_attn is not None
                sa = layer.self_attn
                # 3a. Batched Q (+ optional KV & QSA indexer) projection + Q norm/RoPE across ALL N tokens
                q_packed, pre_k_raw, pre_v_raw, pre_q_idx, pre_raw_k_idx = (
                    sa.project_qkv_and_indexer(
                        attn_in, need_idx=(sa.indexer is not None), need_idx_norm=False
                    )
                )
                if sa.attn_output_gate:
                    q_unnorm, gate_flat = q_packed.unflatten(
                        -1, (sa.num_heads, sa.head_dim * 2)
                    ).chunk(2, dim=-1)
                    gate_4d = gate_flat.unsqueeze(0)
                else:
                    q_unnorm = q_packed.unflatten(-1, (sa.num_heads, sa.head_dim))
                    gate_4d = None
                q_all = gemma_rmsnorm_and_partial_rope(
                    q_unnorm.unsqueeze(0),
                    sa.q_norm.weight,
                    flat_pos_2d,
                    sa.inv_freq,
                    sa.rotary_dim,
                    eps=sa.rms_norm_eps,
                )  # [1, N, H_q, D]

                cu_list = plan.cu_seqlens_host
                if not layer.is_consumer:
                    assert sa.k_proj is not None and sa.v_proj is not None and sa.k_norm is not None
                    k_raw = pre_k_raw if pre_k_raw is not None else dense_linear_bf16_or_tgv(sa.k_proj, attn_in)
                    v_raw = pre_v_raw if pre_v_raw is not None else dense_linear_bf16_or_tgv(sa.v_proj, attn_in)
                    if sa.use_kv_sconv and sa.k_sconv is not None and sa.v_sconv is not None:
                        if kv_cache is not None and FULL_ATTN_KV_SCONV_TAG in layer_caches:
                            sc_c = layer_caches[FULL_ATTN_KV_SCONV_TAG]
                            bt_sc = _active_bt(FULL_ATTN_KV_SCONV_TAG)
                            init_kv_sc = _read_init_state(
                                sc_c,
                                bt_sc,
                                FULL_ATTN_KV_SCONV_TAG,
                                lid,
                                (2, self.cfg.num_key_value_heads * self.cfg.head_dim, sa.kv_sconv_kernel_size - 1),
                                dtype,
                            )
                            s_tok, s_req, s_end, r_sstart, s_phys, _, a32 = _snap_phys_for(sc_c, bt_sc)
                            sc_pool_bf16, _ = _view_opaque_pool_2d(sc_c.kv_cache_base, dtype)
                            kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim
                            sc_len = sa.kv_sconv_kernel_size - 1
                            k_raw = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                                k_raw,
                                sa.k_sconv.weight,
                                plan.cu_seqlens_i32,
                                plan.tile_req_32,
                                plan.tile_t0_32,
                                init_state=init_kv_sc[:, 0] if init_kv_sc is not None else None,
                                snap_tok_idx=s_tok,
                                snap_req_idx=s_req,
                                dst_pool=sc_pool_bf16,
                                snap_dst_idx=s_phys,
                                dst_col_offset=0,
                                req_snap_start=r_sstart,
                                snap_local_ends=s_end,
                                snap_aligned=a32,
                                skip_invalid_phys=True,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                                round_pre_act=True,
                                block_t=32,
                            )
                            v_raw = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                                v_raw,
                                sa.v_sconv.weight,
                                plan.cu_seqlens_i32,
                                plan.tile_req_32,
                                plan.tile_t0_32,
                                init_state=init_kv_sc[:, 1] if init_kv_sc is not None else None,
                                snap_tok_idx=s_tok,
                                snap_req_idx=s_req,
                                dst_pool=sc_pool_bf16,
                                snap_dst_idx=s_phys,
                                dst_col_offset=kv_dim * sc_len,
                                req_snap_start=r_sstart,
                                snap_local_ends=s_end,
                                snap_aligned=a32,
                                skip_invalid_phys=True,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                                round_pre_act=True,
                                block_t=32,
                            )
                        else:
                            k_raw = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                                k_raw,
                                sa.k_sconv.weight,
                                plan.cu_seqlens_i32,
                                plan.tile_req_32,
                                plan.tile_t0_32,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                                round_pre_act=True,
                                block_t=32,
                            )
                            v_raw = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                                v_raw,
                                sa.v_sconv.weight,
                                plan.cu_seqlens_i32,
                                plan.tile_req_32,
                                plan.tile_t0_32,
                                residual=sa.kv_sconv_residual,
                                use_activation=sa.kv_sconv_use_activation,
                                round_pre_act=True,
                                block_t=32,
                            )

                    k_all = gemma_rmsnorm_and_partial_rope(
                        k_raw.unflatten(-1, (sa.num_kv_heads, sa.head_dim)).unsqueeze(0),
                        sa.k_norm.weight,
                        flat_pos_2d,
                        sa.inv_freq,
                        sa.rotary_dim,
                        eps=sa.rms_norm_eps,
                    )
                    v_all = v_raw.unflatten(-1, (sa.num_kv_heads, sa.head_dim)).unsqueeze(0)

                    kv_c = layer_caches.get(FULL_ATTN_KV_TAG) if kv_cache is not None else None
                    bt_kv_act = _active_bt(FULL_ATTN_KV_TAG) if kv_cache is not None else None
                    if kv_c is not None and bt_kv_act is not None:
                        _store_ragged_full_attn_kv_gpu(
                            kv_c,
                            bt_kv_act,
                            plan.tok_req_i32,
                            plan.tok_abs_pos_i32,
                            k_all.squeeze(0),
                            v_all.squeeze(0),
                        )

                    q_idx_all: Optional[torch.Tensor] = None
                    raw_k_idx_all: Optional[torch.Tensor] = None
                    idx_kv_pool: Optional[torch.Tensor] = None
                    bt_idx_kv_act: Optional[torch.Tensor] = None
                    if sa.indexer is not None:
                        idx = sa.indexer
                        if pre_q_idx is not None and pre_raw_k_idx is not None:
                            q_idx_all = pre_q_idx.unsqueeze(0)
                            raw_k_idx_all = pre_raw_k_idx.unsqueeze(0)
                        else:
                            q_idx_all, raw_k_idx_all = idx.project(attn_in.unsqueeze(0))
                        if kv_cache is not None:
                            idx_kv_c = layer_caches[INDEXER_KV_TAG]
                            idx_st_c = layer_caches[INDEXER_STATE_TAG]
                            bt_idx_kv_act = _active_bt(INDEXER_KV_TAG)
                            bt_idx_st_act = _active_bt(INDEXER_STATE_TAG)
                            idx_kv_tpb = int(idx_kv_c.seq_size_per_block)
                            idx_st_tpb = int(idx_st_c.seq_size_per_block)
                            idx_kv_pool = view_indexer_kv_pool(
                                idx_kv_c,
                                kv_tokens_per_block=idx_kv_tpb,
                                compress_ratio=idx.compress_ratio,
                                head_dim=idx.head_dim,
                            )
                            idx_st_pool = view_indexer_state_pool(
                                idx_st_c,
                                compress_ratio=idx.compress_ratio,
                                head_dim=idx.head_dim,
                            )
                            write_qsa_prefill_cache(
                                raw_k_idx_all.squeeze(0),
                                plan.cu_seqlens_i32,
                                plan.start_pos_i64,
                                flat_positions,
                                idx.k_norm_gamma,
                                idx_kv_pool,
                                bt_idx_kv_act,
                                idx_st_pool,
                                bt_idx_st_act,
                                ratio=idx.compress_ratio,
                                kv_tokens_per_block=idx_kv_tpb,
                                state_tokens_per_block=idx_st_tpb,
                                rotary_dim=sa.indexer_rotary_dim,
                                rope_theta=sa.rope_theta,
                                norm_eps=idx.norm_eps,
                                tok_req_i32=plan.tok_req_i32,
                            )
                            source_idx_prefill[lid] = (idx_kv_pool, bt_idx_kv_act)
                        else:
                            full_qsa_raw_reqs[lid] = []

                    # Batched uniform path OR per-request ragged path (once over full sequence!)
                    full_kv_reqs[lid] = []
                    out_flat: Optional[torch.Tensor] = None
                    if uniform_full_batch and (kv_cache is None or kv_c is not None):
                        bsz_u = plan.bsz
                        t_in = plan.active_lens[0]
                        t_pfx = plan.active_pfx[0]
                        t_tot = t_pfx + t_in
                        use_paged_qsa = (
                            sa.indexer is not None
                            and t_pfx > 0
                            and kv_c is not None
                            and bt_kv_act is not None
                        )
                        if kv_c is not None and bt_kv_act is not None:
                            source_kv_prefill[lid] = (kv_c, bt_kv_act)
                        q_u_bhtd = q_all.view(bsz_u, t_in, sa.num_heads, sa.head_dim).transpose(1, 2)
                        if sa.indexer is not None:
                            idx = sa.indexer
                            assert q_idx_all is not None and raw_k_idx_all is not None
                            q_idx_u = q_idx_all.view(bsz_u, t_in, idx.n_heads, idx.head_dim)
                            raw_k_u = raw_k_idx_all.view(bsz_u, t_in, idx.head_dim)
                            pos_u = flat_pos_2d.view(bsz_u, t_in)
                            if kv_cache is None:
                                k_full_b = k_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                                v_full_b = v_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                                full_kv_batched[lid] = (k_full_b, v_full_b)
                                for bi in range(bsz_u):
                                    full_kv_reqs[lid].append(
                                        FullAttnKVCache(
                                            k_cache=k_full_b[bi : bi + 1],
                                            v_cache=v_full_b[bi : bi + 1],
                                            seq_len=t_tot,
                                        )
                                    )
                                full_qsa_raw_batched[lid] = (raw_k_u, idx.k_norm_gamma)
                                for bi in range(bsz_u):
                                    full_qsa_raw_reqs[lid].append((raw_k_u[bi : bi + 1], idx.k_norm_gamma))
                            if t_pfx == 0 or kv_cache is None:
                                k_full_b = k_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                                v_full_b = v_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                                full_kv_batched[lid] = (k_full_b, v_full_b)
                                k_u_bhtd = k_full_b.transpose(1, 2)
                                v_u_bhtd = v_full_b.transpose(1, 2)
                                pref_selected = qsa_select_causal_from_projected(
                                    idx,
                                    q_idx_u,
                                    raw_k_u,
                                    q_norm_gamma=idx.q_norm_gamma,
                                    positions=pos_u,
                                    rotary_dim=sa.indexer_rotary_dim,
                                    rope_theta=sa.rope_theta,
                                    **cp_idx_kw,
                                )
                                out_u = qsa_sparse_prefill_attention(
                                    q_u_bhtd,
                                    k_u_bhtd,
                                    v_u_bhtd,
                                    pref_selected,
                                    assume_unique_selected=(idx.compress_ratio == 4),
                                )
                            else:
                                assert idx_kv_pool is not None and bt_idx_kv_act is not None
                                vis_q = _get_vis_q(bsz_u, t_pfx, t_in)
                                cap_ctx_u = (
                                    prefill_handle.cg_max_ctx_tokens
                                    if (prefill_handle is not None and prefill_handle.cg_max_ctx_tokens is not None)
                                    else t_tot
                                )
                                max_ctx_p = max(1, cap_ctx_u // idx.compress_ratio)
                                pref_selected = score_and_select_qsa_paged(
                                    q_idx_u,
                                    pos_u,
                                    idx_kv_pool,
                                    bt_idx_kv_act[:bsz_u],
                                    vis_q,
                                    compress_ratio=idx.compress_ratio,
                                    token_budget=idx.token_budget,
                                    rotary_dim=sa.indexer_rotary_dim,
                                    rope_theta=sa.rope_theta,
                                    max_ctx_len=max_ctx_p,
                                    is_cuda_graph=bool(prefill_handle is not None and prefill_handle.is_cuda_graph),
                                    q_norm_gamma=idx.q_norm_gamma,
                                    norm_eps=idx.norm_eps,
                                    **cp_idx_kw,
                                )
                                if use_paged_qsa:
                                    assert kv_c is not None and bt_kv_act is not None
                                    out_u = qsa_sparse_paged_prefill_attention(
                                        q_u_bhtd,
                                        kv_c.kv_cache_base,
                                        bt_kv_act[:bsz_u],
                                        pref_selected,
                                        page_size=int(kv_c.seq_size_per_block),
                                        kv_heads=sa.num_kv_heads,
                                        head_dim=sa.head_dim,
                                        max_ctx_len=cap_ctx_u,
                                        kv_scale=getattr(kv_c, "kv_scale_base", None),
                                        assume_unique_selected=(idx.compress_ratio == 4),
                                    )
                                else:
                                    k_full_b, v_full_b = _load_batched_full_attn_kv(
                                        kv_c.kv_cache_base,
                                        bt_kv_act,
                                        bsz_u,
                                        t_tot,
                                        int(kv_c.seq_size_per_block),
                                        num_kv_heads=sa.num_kv_heads,
                                        head_dim=sa.head_dim,
                                        kv_scale=getattr(kv_c, "kv_scale_base", None),
                                        out_dtype=attn_in.dtype,
                                    )
                                    full_kv_batched[lid] = (k_full_b, v_full_b)
                                    out_u = qsa_sparse_prefill_attention(
                                        q_u_bhtd,
                                        k_full_b.transpose(1, 2),
                                        v_full_b.transpose(1, 2),
                                        pref_selected,
                                        assume_unique_selected=(idx.compress_ratio == 4),
                                    )
                        else:
                            if t_pfx > 0 and kv_c is not None and bt_kv_act is not None:
                                k_full_b, v_full_b = _load_batched_full_attn_kv(
                                    kv_c.kv_cache_base,
                                    bt_kv_act,
                                    bsz_u,
                                    t_tot,
                                    int(kv_c.seq_size_per_block),
                                    num_kv_heads=sa.num_kv_heads,
                                    head_dim=sa.head_dim,
                                    kv_scale=getattr(kv_c, "kv_scale_base", None),
                                    out_dtype=attn_in.dtype,
                                )
                            else:
                                k_full_b = k_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                                v_full_b = v_all.view(bsz_u, t_in, sa.num_kv_heads, sa.head_dim)
                            full_kv_batched[lid] = (k_full_b, v_full_b)
                            for bi in range(bsz_u):
                                full_kv_reqs[lid].append(
                                    FullAttnKVCache(
                                        k_cache=k_full_b[bi : bi + 1],
                                        v_cache=v_full_b[bi : bi + 1],
                                        seq_len=t_tot,
                                    )
                                )
                            k_u_bhtd = k_full_b.transpose(1, 2)
                            v_u_bhtd = v_full_b.transpose(1, 2)
                            if t_pfx == 0:
                                out_u = torch.nn.functional.scaled_dot_product_attention(
                                    q_u_bhtd,
                                    k_u_bhtd,
                                    v_u_bhtd,
                                    attn_mask=None,
                                    is_causal=True,
                                    scale=sa.scaling,
                                    enable_gqa=True,
                                )
                            else:
                                out_u = torch.nn.functional.scaled_dot_product_attention(
                                    q_u_bhtd,
                                    k_u_bhtd,
                                    v_u_bhtd,
                                    attn_mask=_get_sdpa_mask(t_pfx, t_tot),
                                    is_causal=False,
                                    scale=sa.scaling,
                                    enable_gqa=True,
                                )
                        gate_u = (
                            gate_4d.view(bsz_u, t_in, sa.num_heads, sa.head_dim)
                            if gate_4d is not None
                            else None
                        )
                        out_flat = apply_attn_output_gate_flatten(out_u, gate_u).reshape(
                            total_tokens, sa.num_heads * sa.head_dim
                        )
                    else:
                        attn_bhtd = torch.empty(
                            (1, sa.num_heads, total_tokens, sa.head_dim),
                            dtype=dtype,
                            device=device,
                        )
                        for bi in range(plan.bsz):
                            s, e = int(cu_list[bi]), int(cu_list[bi + 1])
                            t_in = e - s
                            t_pfx = plan.active_pfx[bi]
                            t_tot = t_pfx + t_in
                            if t_pfx > 0 and kv_c is not None and bt_kv_act is not None:
                                tpb_kv = int(kv_c.seq_size_per_block)
                                k_pfx, v_pfx = _load_single_req_full_attn_kv(
                                    kv_c.kv_cache_base,
                                    bt_kv_act[bi],
                                    t_pfx,
                                    tpb_kv,
                                    num_kv_heads=sa.num_kv_heads,
                                    head_dim=sa.head_dim,
                                    kv_scale=getattr(kv_c, "kv_scale_base", None),
                                    out_dtype=attn_in.dtype,
                                )
                                k_full = torch.cat([k_pfx.unsqueeze(0), k_all[:, s:e]], dim=1)
                                v_full = torch.cat([v_pfx.unsqueeze(0), v_all[:, s:e]], dim=1)
                            else:
                                k_full = k_all[:, s:e]
                                v_full = v_all[:, s:e]
                            full_kv_reqs[lid].append(
                                FullAttnKVCache(k_cache=k_full, v_cache=v_full, seq_len=t_tot)
                            )

                            q_b_bhtd = q_all[:, s:e].transpose(1, 2)
                            k_b_bhtd = k_full.transpose(1, 2)
                            v_b_bhtd = v_full.transpose(1, 2)
                            if sa.indexer is not None:
                                idx = sa.indexer
                                assert q_idx_all is not None and raw_k_idx_all is not None
                                q_idx_b = q_idx_all[:, s:e]
                                raw_k_b = raw_k_idx_all[:, s:e]
                                pos_b = flat_pos_2d[:, s:e]
                                if kv_cache is None:
                                    full_qsa_raw_reqs[lid].append((raw_k_b, idx.k_norm_gamma))
                                if t_pfx == 0 or kv_cache is None:
                                    pref_selected = qsa_select_causal_from_projected(
                                        idx,
                                        q_idx_b,
                                        raw_k_b,
                                        q_norm_gamma=idx.q_norm_gamma,
                                        positions=pos_b,
                                        rotary_dim=sa.indexer_rotary_dim,
                                        rope_theta=sa.rope_theta,
                                        **cp_idx_kw,
                                    )
                                else:
                                    assert idx_kv_pool is not None and bt_idx_kv_act is not None
                                    vis_q = _get_vis_q(1, t_pfx, t_in)
                                    max_ctx_p = max(1, t_tot // idx.compress_ratio)
                                    pref_selected = score_and_select_qsa_paged(
                                        q_idx_b,
                                        pos_b,
                                        idx_kv_pool,
                                        bt_idx_kv_act[bi : bi + 1],
                                        vis_q,
                                        compress_ratio=idx.compress_ratio,
                                        token_budget=idx.token_budget,
                                        rotary_dim=sa.indexer_rotary_dim,
                                        rope_theta=sa.rope_theta,
                                        max_ctx_len=max_ctx_p,
                                        is_cuda_graph=bool(prefill_handle is not None and prefill_handle.is_cuda_graph),
                                        q_norm_gamma=idx.q_norm_gamma,
                                        norm_eps=idx.norm_eps,
                                        **cp_idx_kw,
                                    )
                                out_b = qsa_sparse_prefill_attention(
                                    q_b_bhtd,
                                    k_b_bhtd,
                                    v_b_bhtd,
                                    pref_selected,
                                    assume_unique_selected=(idx.compress_ratio == 4),
                                )
                            else:
                                if t_pfx == 0:
                                    out_b = torch.nn.functional.scaled_dot_product_attention(
                                        q_b_bhtd,
                                        k_b_bhtd,
                                        v_b_bhtd,
                                        attn_mask=None,
                                        is_causal=True,
                                        scale=sa.scaling,
                                        enable_gqa=True,
                                    )
                                else:
                                    out_b = torch.nn.functional.scaled_dot_product_attention(
                                        q_b_bhtd,
                                        k_b_bhtd,
                                        v_b_bhtd,
                                        attn_mask=_get_sdpa_mask(t_pfx, t_tot),
                                        is_causal=False,
                                        scale=sa.scaling,
                                        enable_gqa=True,
                                    )
                            attn_bhtd[:, :, s:e].copy_(out_b)
                else:
                    # Consumer FullAttn / QSA
                    src_lid = layer.kv_cache_owner
                    src_kv_list = full_kv_reqs[src_lid]
                    q_idx_cons: Optional[torch.Tensor] = None
                    if sa.indexer is not None:
                        if pre_q_idx is not None:
                            q_idx_cons = pre_q_idx.unsqueeze(0)
                        else:
                            q_idx_cons, _ = sa.indexer.project_raw(attn_in.unsqueeze(0))

                    out_flat = None
                    if uniform_full_batch and (src_lid in full_kv_batched or src_lid in source_kv_prefill):
                        bsz_u = plan.bsz
                        t_in = plan.active_lens[0]
                        t_pfx = plan.active_pfx[0]
                        t_tot = t_pfx + t_in
                        q_u_bhtd = q_all.view(bsz_u, t_in, sa.num_heads, sa.head_dim).transpose(1, 2)
                        if sa.indexer is not None:
                            idx = sa.indexer
                            assert q_idx_cons is not None
                            q_idx_u = q_idx_cons.view(bsz_u, t_in, idx.n_heads, idx.head_dim)
                            pos_u = flat_pos_2d.view(bsz_u, t_in)
                            if kv_cache is not None:
                                src_idx_kv_pool, src_bt_idx_kv_act = source_idx_prefill[src_lid]
                                vis_q = _get_vis_q(bsz_u, t_pfx, t_in)
                                cap_ctx_u = (
                                    prefill_handle.cg_max_ctx_tokens
                                    if (prefill_handle is not None and prefill_handle.cg_max_ctx_tokens is not None)
                                    else t_tot
                                )
                                max_ctx_p = max(1, cap_ctx_u // idx.compress_ratio)
                                cons_selected = score_and_select_qsa_paged(
                                    q_idx_u,
                                    pos_u,
                                    src_idx_kv_pool,
                                    src_bt_idx_kv_act[:bsz_u],
                                    vis_q,
                                    compress_ratio=idx.compress_ratio,
                                    token_budget=idx.token_budget,
                                    rotary_dim=sa.indexer_rotary_dim,
                                    rope_theta=sa.rope_theta,
                                    max_ctx_len=max_ctx_p,
                                    is_cuda_graph=bool(prefill_handle is not None and prefill_handle.is_cuda_graph),
                                    q_norm_gamma=idx.q_norm_gamma,
                                    norm_eps=idx.norm_eps,
                                    **cp_idx_kw,
                                )
                            else:
                                src_raw_b, src_kn = full_qsa_raw_batched[src_lid]
                                cons_selected = qsa_select_causal_from_projected(
                                    idx,
                                    q_idx_u,
                                    src_raw_b,
                                    k_norm_gamma=src_kn,
                                    q_norm_gamma=idx.q_norm_gamma,
                                    positions=pos_u,
                                    rotary_dim=sa.indexer_rotary_dim,
                                    rope_theta=sa.rope_theta,
                                    **cp_idx_kw,
                                )
                            if t_pfx > 0 and kv_cache is not None and src_lid in source_kv_prefill:
                                src_kv_c, src_bt_act = source_kv_prefill[src_lid]
                                cap_ctx_c = (
                                    prefill_handle.cg_max_ctx_tokens
                                    if (prefill_handle is not None and prefill_handle.cg_max_ctx_tokens is not None)
                                    else t_tot
                                )
                                out_u = qsa_sparse_paged_prefill_attention(
                                    q_u_bhtd,
                                    src_kv_c.kv_cache_base,
                                    src_bt_act[:bsz_u],
                                    cons_selected,
                                    page_size=int(src_kv_c.seq_size_per_block),
                                    kv_heads=sa.num_kv_heads,
                                    head_dim=sa.head_dim,
                                    max_ctx_len=cap_ctx_c,
                                    kv_scale=getattr(src_kv_c, "kv_scale_base", None),
                                    assume_unique_selected=(idx.compress_ratio == 4),
                                )
                            else:
                                k_full_b, v_full_b = _ensure_source_kv_batched(src_lid)
                                out_u = qsa_sparse_prefill_attention(
                                    q_u_bhtd,
                                    k_full_b[:, :t_tot].transpose(1, 2),
                                    v_full_b[:, :t_tot].transpose(1, 2),
                                    cons_selected,
                                    assume_unique_selected=(idx.compress_ratio == 4),
                                )
                        else:
                            k_full_b, v_full_b = _ensure_source_kv_batched(src_lid)
                            k_u_bhtd = k_full_b[:, :t_tot].transpose(1, 2)
                            v_u_bhtd = v_full_b[:, :t_tot].transpose(1, 2)
                            if t_pfx == 0:
                                out_u = torch.nn.functional.scaled_dot_product_attention(
                                    q_u_bhtd,
                                    k_u_bhtd,
                                    v_u_bhtd,
                                    attn_mask=None,
                                    is_causal=True,
                                    scale=sa.scaling,
                                    enable_gqa=True,
                                )
                            else:
                                out_u = torch.nn.functional.scaled_dot_product_attention(
                                    q_u_bhtd,
                                    k_u_bhtd,
                                    v_u_bhtd,
                                    attn_mask=_get_sdpa_mask(t_pfx, t_tot),
                                    is_causal=False,
                                    scale=sa.scaling,
                                    enable_gqa=True,
                                )
                        gate_u = (
                            gate_4d.view(bsz_u, t_in, sa.num_heads, sa.head_dim)
                            if gate_4d is not None
                            else None
                        )
                        out_flat = apply_attn_output_gate_flatten(out_u, gate_u).reshape(
                            total_tokens, sa.num_heads * sa.head_dim
                        )
                    else:
                        attn_bhtd = torch.empty(
                            (1, sa.num_heads, total_tokens, sa.head_dim),
                            dtype=dtype,
                            device=device,
                        )
                        for bi in range(plan.bsz):
                            s, e = int(cu_list[bi]), int(cu_list[bi + 1])
                            t_in = e - s
                            t_pfx = plan.active_pfx[bi]
                            t_tot = t_pfx + t_in
                            kv_obj = src_kv_list[bi]
                            q_b_bhtd = q_all[:, s:e].transpose(1, 2)
                            k_b_bhtd = kv_obj.k_cache[:, :t_tot].transpose(1, 2)
                            v_b_bhtd = kv_obj.v_cache[:, :t_tot].transpose(1, 2)
                            if sa.indexer is not None:
                                idx = sa.indexer
                                assert q_idx_cons is not None
                                q_idx_b = q_idx_cons[:, s:e]
                                pos_b = flat_pos_2d[:, s:e]
                                if kv_cache is not None:
                                    src_idx_kv_pool, src_bt_idx_kv_act = source_idx_prefill[src_lid]
                                    vis_q = _get_vis_q(1, t_pfx, t_in)
                                    max_ctx_p = max(1, t_tot // idx.compress_ratio)
                                    cons_selected = score_and_select_qsa_paged(
                                        q_idx_b,
                                        pos_b,
                                        src_idx_kv_pool,
                                        src_bt_idx_kv_act[bi : bi + 1],
                                        vis_q,
                                        compress_ratio=idx.compress_ratio,
                                        token_budget=idx.token_budget,
                                        rotary_dim=sa.indexer_rotary_dim,
                                        rope_theta=sa.rope_theta,
                                        max_ctx_len=max_ctx_p,
                                        is_cuda_graph=bool(prefill_handle is not None and prefill_handle.is_cuda_graph),
                                        q_norm_gamma=idx.q_norm_gamma,
                                        norm_eps=idx.norm_eps,
                                        **cp_idx_kw,
                                    )
                                else:
                                    src_raw, src_kn = full_qsa_raw_reqs[src_lid][bi]
                                    cons_selected = qsa_select_causal_from_projected(
                                        idx,
                                        q_idx_b,
                                        src_raw,
                                        k_norm_gamma=src_kn,
                                        q_norm_gamma=idx.q_norm_gamma,
                                        positions=pos_b,
                                        rotary_dim=sa.indexer_rotary_dim,
                                        rope_theta=sa.rope_theta,
                                        **cp_idx_kw,
                                    )
                                out_b = qsa_sparse_prefill_attention(
                                    q_b_bhtd,
                                    k_b_bhtd,
                                    v_b_bhtd,
                                    cons_selected,
                                    assume_unique_selected=(idx.compress_ratio == 4),
                                )
                            else:
                                if t_pfx == 0:
                                    out_b = torch.nn.functional.scaled_dot_product_attention(
                                        q_b_bhtd,
                                        k_b_bhtd,
                                        v_b_bhtd,
                                        attn_mask=None,
                                        is_causal=True,
                                        scale=sa.scaling,
                                        enable_gqa=True,
                                    )
                                else:
                                    out_b = torch.nn.functional.scaled_dot_product_attention(
                                        q_b_bhtd,
                                        k_b_bhtd,
                                        v_b_bhtd,
                                        attn_mask=_get_sdpa_mask(t_pfx, t_tot),
                                        is_causal=False,
                                        scale=sa.scaling,
                                        enable_gqa=True,
                                    )
                            attn_bhtd[:, :, s:e].copy_(out_b)

                if out_flat is None:
                    out_flat = apply_attn_output_gate_flatten(attn_bhtd, gate_4d).squeeze(0)
                attn_out = sa._maybe_all_reduce(
                    dense_linear_bf16_or_tgv(sa.o_proj, out_flat), skip_allreduce=sp_active
                )

            if sp_active:
                attn_out = self._sp_reduce_scatter_dim0(
                    attn_out,
                    total_tokens=total_tokens,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                )

            # 4. Attention Output ShortConv + clip (in [N_loc, H] SP domain)
            if layer.attn_output_sconv is not None:
                tag_out = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                if kv_cache is not None:
                    c_ao = layer_caches[tag_out]
                    bt_ao = _active_bt(tag_out)
                    init_ao = _read_init_state(
                        c_ao,
                        bt_ao,
                        tag_out,
                        lid,
                        (self.cfg.hidden_size, layer.attn_output_sconv.kernel_size - 1),
                        dtype,
                    )
                    if sp_active:
                        assert sub_plan is not None
                        _, _, _, _, s_phys_ao_all, _, _ = _snap_phys_for(c_ao, bt_ao)
                        sp_snap_sync_specs.append(
                            (
                                c_ao.kv_cache_base,
                                int(c_ao.seq_size_per_block),
                                bt_ao,
                                self.cfg.hidden_size * max(0, layer.attn_output_sconv.kernel_size - 1) * 2,
                                s_phys_ao_all,
                            )
                        )
                        attn_out = self._run_sp_ragged_sconv(
                            attn_out,
                            layer.attn_output_sconv,
                            plan,
                            sub_plan,
                            init_ao,
                            c_ao,
                            bt_ao,
                            eff_sp,
                            eff_rk,
                            eff_grp,
                            eff_comm,
                            clamp_val=self.cfg.clip_attn_output,
                            s_phys_override=_sp_snap_phys_for(c_ao, bt_ao),
                        )
                    else:
                        s_tok, s_req, s_end, r_sstart, s_phys, _, a32 = _snap_phys_for(c_ao, bt_ao)
                        ao_pool_bf16, _ = _view_opaque_pool_2d(c_ao.kv_cache_base, dtype)
                        attn_out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                            attn_out,
                            layer.attn_output_sconv.weight,
                            plan.cu_seqlens_i32,
                            plan.tile_req_32,
                            plan.tile_t0_32,
                            init_state=init_ao,
                            snap_tok_idx=s_tok,
                            snap_req_idx=s_req,
                            dst_pool=ao_pool_bf16,
                            snap_dst_idx=s_phys,
                            req_snap_start=r_sstart,
                            snap_local_ends=s_end,
                            snap_aligned=a32,
                            skip_invalid_phys=True,
                            residual=layer.attn_output_sconv.residual,
                            use_activation=layer.attn_output_sconv.use_activation,
                            clamp_val=self.cfg.clip_attn_output,
                            block_t=32,
                        )
                elif sp_active:
                    assert sub_plan is not None
                    attn_out = self._run_sp_ragged_sconv(
                        attn_out,
                        layer.attn_output_sconv,
                        plan,
                        sub_plan,
                        None,
                        None,
                        None,
                        eff_sp,
                        eff_rk,
                        eff_grp,
                        eff_comm,
                        clamp_val=self.cfg.clip_attn_output,
                    )
                else:
                    attn_out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                        attn_out,
                        layer.attn_output_sconv.weight,
                        plan.cu_seqlens_i32,
                        plan.tile_req_32,
                        plan.tile_t0_32,
                        residual=layer.attn_output_sconv.residual,
                        use_activation=layer.attn_output_sconv.use_activation,
                        clamp_val=self.cfg.clip_attn_output,
                        block_t=32,
                    )
            elif self.cfg.clip_attn_output is not None and n_loc > 0:
                attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)

            # 5. MLP HC + LatentMoE / DenseMLP (in [N_loc, H] SP domain)
            mlp_in, hyper_hidden, mlp_inj, prev_latent = self._hc_combine_and_mix_safe(
                layer.mlp_hyper_connection, attn_out, hyper_hidden, attn_inj, prev_latent
            )
            assert mlp_inj is not None
            mlp_out = _run_runtime_mlp(
                layer,
                mlp_in,
                sp_input=sp_active,
                sp_total_tokens=total_tokens if sp_active else None,
            )

            # 6. FFN Output ShortConv + clip (in [N_loc, H] SP domain)
            if layer.ffn_output_sconv is not None:
                if kv_cache is not None:
                    c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                    bt_fo = _active_bt(FFN_OUTPUT_SCONV_TAG)
                    init_fo = _read_init_state(
                        c_fo,
                        bt_fo,
                        FFN_OUTPUT_SCONV_TAG,
                        lid,
                        (self.cfg.hidden_size, layer.ffn_output_sconv.kernel_size - 1),
                        dtype,
                    )
                    if sp_active:
                        assert sub_plan is not None
                        _, _, _, _, s_phys_fo_all, _, _ = _snap_phys_for(c_fo, bt_fo)
                        sp_snap_sync_specs.append(
                            (
                                c_fo.kv_cache_base,
                                int(c_fo.seq_size_per_block),
                                bt_fo,
                                self.cfg.hidden_size * max(0, layer.ffn_output_sconv.kernel_size - 1) * 2,
                                s_phys_fo_all,
                            )
                        )
                        mlp_out = self._run_sp_ragged_sconv(
                            mlp_out,
                            layer.ffn_output_sconv,
                            plan,
                            sub_plan,
                            init_fo,
                            c_fo,
                            bt_fo,
                            eff_sp,
                            eff_rk,
                            eff_grp,
                            eff_comm,
                            clamp_val=self.cfg.clip_mlp_output,
                            s_phys_override=_sp_snap_phys_for(c_fo, bt_fo),
                        )
                    else:
                        s_tok, s_req, s_end, r_sstart, s_phys, _, a32 = _snap_phys_for(c_fo, bt_fo)
                        fo_pool_bf16, _ = _view_opaque_pool_2d(c_fo.kv_cache_base, dtype)
                        mlp_out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                            mlp_out,
                            layer.ffn_output_sconv.weight,
                            plan.cu_seqlens_i32,
                            plan.tile_req_32,
                            plan.tile_t0_32,
                            init_state=init_fo,
                            snap_tok_idx=s_tok,
                            snap_req_idx=s_req,
                            dst_pool=fo_pool_bf16,
                            snap_dst_idx=s_phys,
                            req_snap_start=r_sstart,
                            snap_local_ends=s_end,
                            snap_aligned=a32,
                            skip_invalid_phys=True,
                            residual=layer.ffn_output_sconv.residual,
                            use_activation=layer.ffn_output_sconv.use_activation,
                            clamp_val=self.cfg.clip_mlp_output,
                            block_t=32,
                        )
                elif sp_active:
                    assert sub_plan is not None
                    mlp_out = self._run_sp_ragged_sconv(
                        mlp_out,
                        layer.ffn_output_sconv,
                        plan,
                        sub_plan,
                        None,
                        None,
                        None,
                        eff_sp,
                        eff_rk,
                        eff_grp,
                        eff_comm,
                        clamp_val=self.cfg.clip_mlp_output,
                    )
                else:
                    mlp_out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                        mlp_out,
                        layer.ffn_output_sconv.weight,
                        plan.cu_seqlens_i32,
                        plan.tile_req_32,
                        plan.tile_t0_32,
                        residual=layer.ffn_output_sconv.residual,
                        use_activation=layer.ffn_output_sconv.use_activation,
                        clamp_val=self.cfg.clip_mlp_output,
                        block_t=32,
                    )
            elif self.cfg.clip_mlp_output is not None and n_loc > 0:
                mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)

            deferred = (mlp_out, mlp_inj)

            if writers and not sp_active:
                publish_qwen4_5_layer_caches(kv_cache, lid, writers)

        assert deferred is not None
        if sp_active:
            if sp_snap_sync_specs:
                self._sync_sp_prefill_snapshots(
                    sp_snap_sync_specs,
                    plan,
                    sync_all_blocks=(
                        bool(prefill_handle.sp_sync_all_blocks_override)
                        if (prefill_handle is not None and prefill_handle.sp_sync_all_blocks_override is not None)
                        else bool(writers)
                    ),
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                    device=device,
                )
            if writers:
                for lid_p in range(len(model.layers)):
                    publish_qwen4_5_layer_caches(kv_cache, lid_p, writers)
        if n_loc == 0:
            flat_out = torch.empty((0, self.cfg.hidden_size), dtype=dtype, device=device)
        else:
            flat_out, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
                deferred[0], hyper_hidden, deferred[1], prev_latent=None
            )
        if sp_active:
            flat_out = self._sp_all_gather_dim0(
                flat_out,
                total_tokens=total_tokens,
                sp_size=eff_sp,
                sp_rank=eff_rk,
                sp_group=eff_grp,
                sp_comm=eff_comm,
            )
            if aux_out is not None:
                aux_out = self._sp_all_gather_dim0(
                    aux_out,
                    total_tokens=total_tokens,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                )
        if aux_out is not None:
            return flat_out, aux_out
        return flat_out

    def _run_target_verify_batch_gpu(
        self,
        model: Qwen4_5Model,
        input_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
        *,
        seq_lens_before_dev: torch.Tensor,
        verify_handle: Qwen4_5TargetVerifyHandle,
        tx_stage: "_Qwen4_5OpaqueTxStage",
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """CUDA-graph-safe fused batched [B, V] target verification with static staged OPAQUE_STATE."""
        with _model_a2a_step_scope(model):
            return self._run_target_verify_batch_gpu_impl(
                model,
                input_ids_2d,
                positions_2d,
                kv_cache,
                attention_inputs,
                seq_lens_before_dev=seq_lens_before_dev,
                verify_handle=verify_handle,
                tx_stage=tx_stage,
                enable_yoco_decode=enable_yoco_decode,
                capture_layer_ids=capture_layer_ids,
            )

    def _run_target_verify_batch_gpu_impl(
        self,
        model: Qwen4_5Model,
        input_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
        *,
        seq_lens_before_dev: torch.Tensor,
        verify_handle: Qwen4_5TargetVerifyHandle,
        tx_stage: "_Qwen4_5OpaqueTxStage",
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = input_ids_2d.device
        dtype = model.embed_tokens.weight.dtype
        bsz, verify_w = int(input_ids_2d.shape[0]), int(input_ids_2d.shape[1])
        h_size = int(self.cfg.hidden_size)
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )

        eff_sp_e, _, _, _ = self._resolve_sp_dist()
        if bsz == 0 or verify_w == 0:
            self._run_empty_ep_collectives(model, dtype, device, sp_active=(eff_sp_e > 1 and verify_w >= eff_sp_e))
            empty_h = torch.empty((0, h_size), dtype=dtype, device=device)
            if capture_layer_ids:
                return empty_h, torch.empty((0, len(capture_layer_ids) * h_size), dtype=dtype, device=device)
            return empty_h

        tx_stage.begin_forward(seq_lens_before_dev, kv_cache, attention_inputs)

        eff_sp, eff_rk, eff_grp, eff_comm = self._resolve_sp_dist()
        if eff_sp > 1 and bsz >= eff_sp and (bsz % eff_sp) == 0:
            from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

            vrf_sp_mode = "batch"
            b0, b1 = sp_slice_bounds(bsz, eff_sp, eff_rk)
            v0, v1 = 0, verify_w
            loc_bsz, loc_v = b1 - b0, verify_w
            ids_loc_2d = input_ids_2d[b0:b1]
            seq_loc_dev = seq_lens_before_dev[b0:b1]
        elif eff_sp > 1 and bsz == 1 and verify_w >= eff_sp and (verify_w % eff_sp) == 0:
            from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

            vrf_sp_mode = "step"
            b0, b1 = 0, 1
            v0, v1 = sp_slice_bounds(verify_w, eff_sp, eff_rk)
            loc_bsz, loc_v = 1, v1 - v0
            ids_loc_2d = input_ids_2d[:, v0:v1]
            seq_loc_dev = seq_lens_before_dev
        else:
            vrf_sp_mode = "none"
            b0, b1, v0, v1 = 0, bsz, 0, verify_w
            loc_bsz, loc_v = bsz, verify_w
            ids_loc_2d = input_ids_2d
            seq_loc_dev = seq_lens_before_dev
        vrf_sp_active = vrf_sp_mode != "none"
        tx_stage.sp_mode = vrf_sp_mode if vrf_sp_active else None
        tx_stage.sp_bounds = (b0, b1) if vrf_sp_mode == "batch" else ((v0, v1) if vrf_sp_mode == "step" else None)

        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None
        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = tx_stage.get_or_alloc_aux(
                loc_bsz * loc_v, len(capture_layer_ids) * h_size, dtype=dtype, device=device
            )

        emb = model.embed_tokens(ids_loc_2d)  # [B_loc, V_loc, H]
        hyper_hidden = (
            emb
            if self.cfg.hc_count > 1 and (len(model.layers) == 0 or model.layers[0].ple is None)
            else emb.repeat(1, 1, self.cfg.hc_count)
        )
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn_batched: Dict[int, SharedGDNInputs] = {}
        full_kv_layer_caches: Dict[int, Any] = {}
        source_idx_caches: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        vrf_vis_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        cp_idx_kw = self._cp_indexer_kwargs()
        steps_i32 = tx_stage.get_steps_arange(verify_w, device)
        vrf_pos_steps_2d: Optional[torch.Tensor] = None

        def _get_vrf_vis(bt_chk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal vrf_pos_steps_2d
            k_p = bt_chk.data_ptr()
            hit = vrf_vis_cache.get(k_p)
            if hit is not None:
                return hit
            if vrf_pos_steps_2d is None:
                vrf_pos_steps_2d = seq_lens_before_dev.unsqueeze(1) + steps_i32 + 1
            act = (bt_chk != 0).any(dim=1)
            vis = torch.where(act.unsqueeze(1), vrf_pos_steps_2d, 0)
            vrf_vis_cache[k_p] = (act, vis)
            return act, vis

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}

            # 1. Optional PLE (batched V-step hash + 1 gather/AllReduce + 1 W_kv GEMM + V-step conv)
            if layer.ple is not None:
                if deferred is not None:
                    hyper_hidden = layer.attn_hyper_connection.combine(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                    deferred = None
                ple_st_cache = layer_caches[PLE_STATE_TAG]
                ple_ctx_cache = layer_caches[PLE_NGRAM_CTX_TAG]
                bt_st = _block_table_for_tag(attention_inputs, PLE_STATE_TAG)
                bt_ctx = _block_table_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
                bt_st_loc = bt_st[b0:b1] if vrf_sp_mode == "batch" else bt_st
                bt_ctx_loc = bt_ctx[b0:b1] if vrf_sp_mode == "batch" else bt_ctx
                conv_shape = (
                    self.cfg.hc_count * self.cfg.hidden_size,
                    max(0, self.cfg.ple_conv_kernel_size - 1),
                )
                ctx_shape = (max(0, self.cfg.ngram_size - 1),)
                init_ctx = self._read_opaque_batch_static_gpu(
                    ple_ctx_cache,
                    bt_ctx_loc,
                    seq_loc_dev,
                    tx_stage.scratch_tensor(("init", PLE_NGRAM_CTX_TAG, lid), (loc_bsz, *ctx_shape), torch.int64, device),
                    PLE_NGRAM_CTX_TAG,
                    lid,
                    fill_value=self.cfg.eos_token_id,
                )
                init_conv = self._read_opaque_batch_static_gpu(
                    ple_st_cache,
                    bt_st_loc,
                    seq_loc_dev,
                    tx_stage.scratch_tensor(("init", PLE_STATE_TAG, lid), (loc_bsz, *conv_shape), dtype, device),
                    PLE_STATE_TAG,
                    lid,
                )
                stage_st = tx_stage.stage_tensor(
                    PLE_STATE_TAG, lid, ple_st_cache, bt_st, conv_shape, dtype
                )
                stage_ctx = tx_stage.stage_tensor(
                    PLE_NGRAM_CTX_TAG, lid, ple_ctx_cache, bt_ctx, ctx_shape, torch.int64
                )
                stage_ctx_loc = stage_ctx[b0:b1] if vrf_sp_mode == "batch" else (stage_ctx[:1, v0:v1] if vrf_sp_mode == "step" else stage_ctx[:bsz])
                stage_st_loc = stage_st[b0:b1] if vrf_sp_mode == "batch" else (stage_st[:1, v0:v1] if vrf_sp_mode == "step" else stage_st[:bsz])
                hyper_hidden = layer.ple.forward_verify(
                    hyper_hidden,
                    ids_loc_2d,
                    init_ctx,
                    init_conv,
                    stage_ctx=stage_ctx_loc,
                    stage_conv=stage_st_loc,
                    sp_tokens_sharded=vrf_sp_active,
                    sp_total_tokens=(bsz * verify_w) if vrf_sp_active else None,
                    sp_halo_exchange=(vrf_sp_mode == "step"),
                    sp_halo_step_start=v0,
                    sp_full_input_ids_2d=input_ids_2d if vrf_sp_mode == "step" else None,
                    sp_size=eff_sp if vrf_sp_active else None,
                    sp_rank=eff_rk if vrf_sp_active else None,
                    sp_group=eff_grp if vrf_sp_active else None,
                    sp_fast_comm=eff_comm if vrf_sp_active else None,
                )

            # 2. Attn Hyper-Connection (1 batched pass across B*V tokens)
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.mix_with_latent(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    layer.attn_hyper_connection.combine_and_mix_with_latent(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
                deferred = None
            assert attn_inj is not None

            if aux_out is not None and lid in cap_slots:
                attn_flat = attn_in.reshape(loc_bsz * loc_v, h_size)
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * h_size : (slot_i + 1) * h_size].copy_(attn_flat)

            if vrf_sp_active:
                attn_in = self._sp_all_gather_dim0(
                    attn_in.reshape(loc_bsz * loc_v, h_size),
                    total_tokens=bsz * verify_w,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                ).view(bsz, verify_w, h_size)

            # 3. Attention (Batched V-step GDN or Batched V-step QSA Sparse)
            if layer.layer_type == "linear_attention":
                assert layer.linear_attn is not None
                ssm_n = math.prod(self.ssm_shape)
                if layer.is_consumer:
                    conv_n_c = math.prod(self.gdn_consumer_conv_shape)
                    init_q_conv: Optional[torch.Tensor] = None
                    init_q_ssm: Optional[torch.Tensor] = None
                    stage_q_conv: Optional[torch.Tensor] = None
                    stage_q_ssm: Optional[torch.Tensor] = None
                    if GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        init_q_ssm = self._read_opaque_batch_static_gpu(
                            c_ssm,
                            bt_ssm,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", GDN_SSM_STATE_TAG, lid), (bsz, *self.ssm_shape), torch.float32, device),
                            GDN_SSM_STATE_TAG,
                            lid,
                        )
                        stage_q_ssm = tx_stage.stage_tensor(
                            GDN_SSM_STATE_TAG, lid, c_ssm, bt_ssm, self.ssm_shape, torch.float32
                        )[:bsz]
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_q = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            init_q_conv = self._read_opaque_batch_static_gpu(
                                c_conv,
                                bt_q,
                                seq_lens_before_dev,
                                tx_stage.scratch_tensor(("init", GDN_CONV_STATE_TAG, lid), (bsz, *self.gdn_consumer_conv_shape), dtype, device),
                                GDN_CONSUMER_Q_CONV_TAG,
                                lid,
                            )
                            stage_q_conv = tx_stage.stage_tensor(
                                GDN_CONV_STATE_TAG, lid, c_conv, bt_q, self.gdn_consumer_conv_shape, dtype
                            )[:bsz]
                    elif GDN_CONSUMER_Q_CONV_TAG in layer_caches:
                        c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                        bt_q = _block_table_for_tag(attention_inputs, GDN_CONSUMER_Q_CONV_TAG)
                        init_q_conv = self._read_opaque_batch_static_gpu(
                            c_q,
                            bt_q,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", GDN_CONSUMER_Q_CONV_TAG, lid), (bsz, *self.gdn_consumer_conv_shape), dtype, device),
                            GDN_CONSUMER_Q_CONV_TAG,
                            lid,
                        )
                        stage_q_conv = tx_stage.stage_tensor(
                            GDN_CONSUMER_Q_CONV_TAG, lid, c_q, bt_q, self.gdn_consumer_conv_shape, dtype
                        )[:bsz]
                        if not yoco_on:
                            z_ssm = tx_stage.scratch_tensor(("zero_ssm", lid), (bsz, *self.ssm_shape), torch.float32, device)
                            z_ssm.zero_()
                            init_q_ssm = z_ssm
                    else:
                        lin_c = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin_c = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        lin_row_elems = int(lin_c.kv_cache_base.shape[1])
                        init_packed = self._read_opaque_batch_static_gpu(
                            lin_c,
                            bt_lin_c,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", LINEAR_ATTN_TAG, lid), (bsz, lin_row_elems), torch.float32, device),
                            LINEAR_ATTN_TAG,
                            lid,
                        )
                        init_q_ssm = init_packed[:, :ssm_n].view(bsz, *self.ssm_shape)
                        init_q_conv = (
                            init_packed[:, ssm_n:]
                            .view(torch.bfloat16)[:, :conv_n_c]
                            .view(bsz, *self.gdn_consumer_conv_shape)
                        )
                        stage_lin = tx_stage.stage_tensor(
                            LINEAR_ATTN_TAG, lid, lin_c, bt_lin_c, (lin_row_elems,), torch.float32
                        )[:bsz]
                        stage_q_ssm = stage_lin[:, :, :ssm_n].view(bsz, verify_w, *self.ssm_shape)
                        stage_q_conv = (
                            stage_lin[:, :, ssm_n:]
                            .view(torch.bfloat16)[:, :, :conv_n_c]
                            .view(bsz, verify_w, *self.gdn_consumer_conv_shape)
                        )

                    attn_out, _ = layer.linear_attn.forward_verify(
                        attn_in,
                        init_conv_state=init_q_conv,
                        init_ssm_state=init_q_ssm,
                        stage_conv_state=stage_q_conv,
                        stage_ssm_state=stage_q_ssm,
                        shared=shared_gdn_batched[layer.source_layer_idx],  # type: ignore[index]
                        enable_yoco_decode=yoco_on,
                        skip_allreduce=vrf_sp_active,
                    )
                else:
                    conv_n = math.prod(self.gdn_conv_shape)
                    stage_conv: Optional[torch.Tensor] = None
                    stage_ssm: Optional[torch.Tensor] = None
                    lazy_kvgb: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
                    gdn = layer.linear_attn
                    use_lazy_gdn = (
                        _HAS_TRITON
                        and attn_in.is_cuda
                        and not (gdn.publishes_shared and yoco_on)
                        and gdn.head_k_dim == gdn.head_v_dim
                        and (gdn.head_k_dim & (gdn.head_k_dim - 1)) == 0
                        and gdn.num_v_heads % gdn.num_k_heads == 0
                    )
                    if GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        init_ssm_buf = self._read_opaque_batch_static_gpu(
                            c_ssm,
                            bt_ssm,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", GDN_SSM_STATE_TAG, lid), (bsz, *self.ssm_shape), torch.float32, device),
                            GDN_SSM_STATE_TAG,
                            lid,
                        )
                        init_ssm = init_ssm_buf
                        if use_lazy_gdn:
                            lk, lv, lg, lb, _ = tx_stage.register_lazy_gdn(
                                GDN_SSM_STATE_TAG,
                                lid,
                                c_ssm,
                                bt_ssm,
                                init_ssm_buf.view(bsz, ssm_n),
                                ssm_n=ssm_n,
                                num_k_heads=gdn.num_k_heads,
                                num_v_heads=gdn.num_v_heads,
                                head_k_dim=gdn.head_k_dim,
                                head_v_dim=gdn.head_v_dim,
                                kv_dtype=dtype,
                            )
                            lazy_kvgb = (lk, lv, lg, lb)
                        else:
                            stage_ssm = tx_stage.stage_tensor(
                                GDN_SSM_STATE_TAG, lid, c_ssm, bt_ssm, self.ssm_shape, torch.float32
                            )[:bsz]
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_conv = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            init_conv = self._read_opaque_batch_static_gpu(
                                c_conv,
                                bt_conv,
                                seq_lens_before_dev,
                                tx_stage.scratch_tensor(("init", GDN_CONV_STATE_TAG, lid), (bsz, *self.gdn_conv_shape), dtype, device),
                                GDN_CONV_STATE_TAG,
                                lid,
                            )
                            stage_conv = tx_stage.stage_tensor(
                                GDN_CONV_STATE_TAG, lid, c_conv, bt_conv, self.gdn_conv_shape, dtype
                            )[:bsz]
                        else:
                            init_conv = tx_stage.scratch_tensor(("zero_conv", lid), (bsz, *self.gdn_conv_shape), dtype, device)
                            init_conv.zero_()
                    else:
                        lin_c = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        lin_row_elems = int(lin_c.kv_cache_base.shape[1])
                        init_packed = self._read_opaque_batch_static_gpu(
                            lin_c,
                            bt_lin,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", LINEAR_ATTN_TAG, lid), (bsz, lin_row_elems), torch.float32, device),
                            LINEAR_ATTN_TAG,
                            lid,
                        )
                        init_ssm = init_packed[:, :ssm_n].view(bsz, *self.ssm_shape)
                        init_conv = (
                            init_packed[:, ssm_n:]
                            .view(torch.bfloat16)[:, :conv_n]
                            .view(bsz, *self.gdn_conv_shape)
                        )
                        if use_lazy_gdn and (conv_n * 2) % 4 == 0:
                            lk, lv, lg, lb, stage_conv_full = tx_stage.register_lazy_gdn(
                                LINEAR_ATTN_TAG,
                                lid,
                                lin_c,
                                bt_lin,
                                init_packed[:, :ssm_n],
                                ssm_n=ssm_n,
                                num_k_heads=gdn.num_k_heads,
                                num_v_heads=gdn.num_v_heads,
                                head_k_dim=gdn.head_k_dim,
                                head_v_dim=gdn.head_v_dim,
                                kv_dtype=dtype,
                                needs_conv_stage=True,
                                conv_shape=self.gdn_conv_shape,
                                conv_i32_elems=(conv_n + 1) // 2,
                            )
                            lazy_kvgb = (lk, lv, lg, lb)
                            stage_conv = stage_conv_full[:bsz] if stage_conv_full is not None else None
                        else:
                            stage_lin = tx_stage.stage_tensor(
                                LINEAR_ATTN_TAG, lid, lin_c, bt_lin, (lin_row_elems,), torch.float32
                            )[:bsz]
                            stage_ssm = stage_lin[:, :, :ssm_n].view(bsz, verify_w, *self.ssm_shape)
                            stage_conv = (
                                stage_lin[:, :, ssm_n:]
                                .view(torch.bfloat16)[:, :, :conv_n]
                                .view(bsz, verify_w, *self.gdn_conv_shape)
                            )

                    attn_out, pub = gdn.forward_verify(
                        attn_in,
                        init_conv_state=init_conv,
                        init_ssm_state=init_ssm,
                        stage_conv_state=stage_conv,
                        stage_ssm_state=stage_ssm,
                        shared=None,
                        enable_yoco_decode=yoco_on,
                        lazy_kvgb_buffers=lazy_kvgb,
                        skip_allreduce=vrf_sp_active,
                    )
                    if pub is not None:
                        shared_gdn_batched[lid] = pub
            else:
                assert layer.self_attn is not None
                sa = layer.self_attn
                if sa.indexer is not None:
                    # Batched V-step QSA Sparse Verify (ZERO step loops!)
                    idx = sa.indexer
                    q_packed, pre_k_raw, pre_v_raw, pre_q_idx, pre_raw_k_idx = (
                        sa.project_qkv_and_indexer(attn_in, need_idx=True, need_idx_norm=False)
                    )
                    if sa.attn_output_gate:
                        q_unnorm, gate_flat = q_packed.unflatten(
                            -1, (sa.num_heads, sa.head_dim * 2)
                        ).chunk(2, dim=-1)
                        gate_4d = gate_flat
                    else:
                        q_unnorm = q_packed.unflatten(-1, (sa.num_heads, sa.head_dim))
                        gate_4d = None
                    q_all = gemma_rmsnorm_and_partial_rope(
                        q_unnorm,
                        sa.q_norm.weight,
                        positions_2d,
                        sa.inv_freq,
                        sa.rotary_dim,
                        eps=sa.rms_norm_eps,
                    )

                    if not layer.is_consumer:
                        assert sa.k_proj is not None and sa.v_proj is not None and sa.k_norm is not None
                        kv_c = layer_caches[FULL_ATTN_KV_TAG]
                        sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                        bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                        active_kv, _ = _get_vrf_vis(bt_kv)

                        k_raw = pre_k_raw if pre_k_raw is not None else dense_linear_bf16_or_tgv(sa.k_proj, attn_in)
                        v_raw = pre_v_raw if pre_v_raw is not None else dense_linear_bf16_or_tgv(sa.v_proj, attn_in)
                        if sc_c is not None and sa.use_kv_sconv and sa.k_sconv is not None and sa.v_sconv is not None:
                            bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
                            sc_shape = (
                                2,
                                self.cfg.num_key_value_heads * self.cfg.head_dim,
                                sa.kv_sconv_kernel_size - 1,
                            )
                            stage_kv_sc = tx_stage.stage_tensor(
                                FULL_ATTN_KV_SCONV_TAG, lid, sc_c, bt_sc, sc_shape, dtype
                            )[:bsz]
                            kv_dim_sc = self.cfg.num_key_value_heads * self.cfg.head_dim
                            kv_pool_2d = self._try_view_sconv_opaque_pool_2d(
                                sc_c, bt_sc, verify_w, sa.kv_sconv_kernel_size, kv_dim_sc * 2 * (sa.kv_sconv_kernel_size - 1), dtype, k_raw
                            )
                            if kv_pool_2d is not None:
                                tpb_sc = int(sc_c.seq_size_per_block)
                                k_raw = causal_dwconv1d_verify_opaque_triton(
                                    k_raw,
                                    sa.k_sconv.weight,
                                    kv_pool_2d,
                                    bt_sc,
                                    seq_lens_before_dev,
                                    stage_kv_sc[:, :, 0],
                                    page_size=tpb_sc,
                                    pool_col_offset=0,
                                    residual=sa.kv_sconv_residual,
                                    use_activation=sa.kv_sconv_use_activation,
                                    round_pre_act=True,
                                )
                                v_raw = causal_dwconv1d_verify_opaque_triton(
                                    v_raw,
                                    sa.v_sconv.weight,
                                    kv_pool_2d,
                                    bt_sc,
                                    seq_lens_before_dev,
                                    stage_kv_sc[:, :, 1],
                                    page_size=tpb_sc,
                                    pool_col_offset=kv_dim_sc * (sa.kv_sconv_kernel_size - 1),
                                    residual=sa.kv_sconv_residual,
                                    use_activation=sa.kv_sconv_use_activation,
                                    round_pre_act=True,
                                )
                            else:
                                kv_sc_buf = self._read_opaque_batch_static_gpu(
                                    sc_c,
                                    bt_sc,
                                    seq_lens_before_dev,
                                    tx_stage.scratch_tensor(("init", FULL_ATTN_KV_SCONV_TAG, lid), (bsz, *sc_shape), dtype, device),
                                    FULL_ATTN_KV_SCONV_TAG,
                                    lid,
                                )
                                k_raw = causal_dwconv1d_verify_triton(
                                    k_raw,
                                    sa.k_sconv.weight,
                                    kv_sc_buf[:, 0],
                                    stage_kv_sc[:, :, 0],
                                    residual=sa.kv_sconv_residual,
                                    use_activation=sa.kv_sconv_use_activation,
                                    round_pre_act=True,
                                )
                                v_raw = causal_dwconv1d_verify_triton(
                                    v_raw,
                                    sa.v_sconv.weight,
                                    kv_sc_buf[:, 1],
                                    stage_kv_sc[:, :, 1],
                                    residual=sa.kv_sconv_residual,
                                    use_activation=sa.kv_sconv_use_activation,
                                    round_pre_act=True,
                                )

                        k_all = gemma_rmsnorm_and_partial_rope(
                            k_raw.unflatten(-1, (sa.num_kv_heads, sa.head_dim)),
                            sa.k_norm.weight,
                            positions_2d,
                            sa.inv_freq,
                            sa.rotary_dim,
                            eps=sa.rms_norm_eps,
                        )
                        v_all = v_raw.unflatten(-1, (sa.num_kv_heads, sa.head_dim))
                        _store_verify_full_attn_kv_gpu(kv_c, bt_kv, seq_lens_before_dev, k_all, v_all)

                        idx_kv_c = layer_caches[INDEXER_KV_TAG]
                        idx_st_c = layer_caches[INDEXER_STATE_TAG]
                        bt_idx_kv = _block_table_for_tag(attention_inputs, INDEXER_KV_TAG)
                        bt_idx_st = _block_table_for_tag(attention_inputs, INDEXER_STATE_TAG)
                        idx_kv_tpb = int(idx_kv_c.seq_size_per_block)
                        idx_st_tpb = int(idx_st_c.seq_size_per_block)
                        idx_kv_pool = view_indexer_kv_pool(
                            idx_kv_c,
                            kv_tokens_per_block=idx_kv_tpb,
                            compress_ratio=idx.compress_ratio,
                            head_dim=idx.head_dim,
                        )
                        idx_st_shape = (2 * idx.compress_ratio, idx.head_dim)
                        init_idx_st = self._read_opaque_batch_static_gpu(
                            idx_st_c,
                            bt_idx_st,
                            seq_lens_before_dev,
                            tx_stage.scratch_tensor(("init", INDEXER_STATE_TAG, lid), (bsz, *idx_st_shape), torch.float32, device),
                            INDEXER_STATE_TAG,
                            lid,
                        )
                        stage_idx_st = tx_stage.stage_tensor(
                            INDEXER_STATE_TAG, lid, idx_st_c, bt_idx_st, idx_st_shape, torch.float32
                        )[:bsz]
                        active_idx, visible_lens = _get_vrf_vis(bt_idx_st)
                        if pre_q_idx is not None and pre_raw_k_idx is not None:
                            q_idx, raw_k_idx = pre_q_idx, pre_raw_k_idx
                        else:
                            q_idx, raw_k_idx = idx.project_raw(attn_in)
                        write_qsa_verify_cache_gpu_(
                            raw_k_idx,
                            seq_lens_before_dev,
                            idx.k_norm_gamma,
                            idx_kv_pool,
                            bt_idx_kv,
                            init_idx_st,
                            stage_idx_st,
                            kv_tokens_per_block=idx_kv_tpb,
                            state_tokens_per_block=idx_st_tpb,
                            ratio=idx.compress_ratio,
                            rotary_dim=sa.indexer_rotary_dim,
                            rope_theta=sa.rope_theta,
                            norm_eps=idx.norm_eps,
                            valid=active_idx,
                        )
                        full_kv_layer_caches[lid] = (kv_c, bt_kv, active_kv)
                        source_idx_caches[lid] = (idx_kv_pool, bt_idx_kv)

                        selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            idx_kv_pool,
                            bt_idx_kv,
                            visible_lens,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=sa.indexer_rotary_dim,
                            rope_theta=sa.rope_theta,
                            max_ctx_len=None,
                            is_cuda_graph=verify_handle.is_cuda_graph,
                            q_norm_gamma=idx.q_norm_gamma,
                            norm_eps=idx.norm_eps,
                            **cp_idx_kw,
                        )
                        attn_bhsd = verify_handle.forward_qsa_batch(
                            q_all.transpose(1, 2),
                            kv_c,
                            bt_kv=bt_kv,
                            kv_lens_2d=visible_lens,
                            selected_3d=selected,
                        )
                    else:
                        src_lid = layer.kv_cache_owner
                        src_kv_c, src_bt_kv, src_active_kv = full_kv_layer_caches[src_lid]
                        src_idx_kv_pool, src_bt_idx_kv = source_idx_caches[src_lid]
                        if pre_q_idx is not None:
                            q_idx = pre_q_idx
                        else:
                            q_idx, _ = idx.project_raw(attn_in)
                        _, visible_lens = _get_vrf_vis(src_bt_kv)
                        selected = score_and_select_qsa_paged(
                            q_idx,
                            positions_2d,
                            src_idx_kv_pool,
                            src_bt_idx_kv,
                            visible_lens,
                            compress_ratio=idx.compress_ratio,
                            token_budget=idx.token_budget,
                            rotary_dim=sa.indexer_rotary_dim,
                            rope_theta=sa.rope_theta,
                            max_ctx_len=None,
                            is_cuda_graph=verify_handle.is_cuda_graph,
                            q_norm_gamma=idx.q_norm_gamma,
                            norm_eps=idx.norm_eps,
                            **cp_idx_kw,
                        )
                        attn_bhsd = verify_handle.forward_qsa_batch(
                            q_all.transpose(1, 2),
                            src_kv_c,
                            bt_kv=src_bt_kv,
                            kv_lens_2d=visible_lens,
                            selected_3d=selected,
                        )

                    out_flat = apply_attn_output_gate_flatten(attn_bhsd, gate_4d)
                    attn_out = sa._maybe_all_reduce(
                        dense_linear_bf16_or_tgv(sa.o_proj, out_flat), skip_allreduce=vrf_sp_active
                    )
                else:
                    # Legacy dense MQA attention path (preproject Q+K+V across V steps in 1 GEMM)
                    pre_vrf = sa.project_qkv_and_indexer(attn_in, need_idx=False)
                    attn_out_steps = []
                    if not layer.is_consumer:
                        kv_c = layer_caches[FULL_ATTN_KV_TAG]
                        sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                        bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                        active_kv, _ = _get_vrf_vis(bt_kv)
                        k_sc_st: Optional[ShortConvState] = None
                        v_sc_st: Optional[ShortConvState] = None
                        stage_kv_sc = None
                        if sc_c is not None:
                            bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
                            kv_sc_buf = self._read_opaque_batch_gpu(
                                sc_c, bt_sc, seq_lens_before_dev, FULL_ATTN_KV_SCONV_TAG, lid, target_dtype=dtype
                            )
                            k_sc_st = ShortConvState(kv_sc_buf[:, 0].clone())
                            v_sc_st = ShortConvState(kv_sc_buf[:, 1].clone())
                            stage_kv_sc = tx_stage.stage_tensor(
                                FULL_ATTN_KV_SCONV_TAG, lid, sc_c, bt_sc, kv_sc_buf.shape[1:], dtype
                            )
                        full_kv_layer_caches[lid] = (kv_c, bt_kv, active_kv)
                        for step in range(verify_w):
                            pos_step_dev = seq_lens_before_dev + step
                            pos_step_2d = positions_2d[:, step : step + 1]
                            step_in = attn_in[:, step : step + 1]
                            visible_lens = torch.where(active_kv, pos_step_dev + 1, 0)

                            def _dense_step_fn(
                                q_3d: torch.Tensor,
                                kv_l: Any,
                                _bt: torch.Tensor = bt_kv,
                                _vl: torch.Tensor = visible_lens,
                                _act: torch.Tensor = active_kv,
                            ) -> torch.Tensor:
                                return verify_handle.forward_dense_step(
                                    q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, active_rows=_act
                                )

                            pre_step = (
                                pre_vrf[0][:, step : step + 1],
                                pre_vrf[1][:, step : step + 1] if pre_vrf[1] is not None else None,
                                pre_vrf[2][:, step : step + 1] if pre_vrf[2] is not None else None,
                                None,
                                None,
                            )
                            a_step = sa.forward_paged_decode(
                                step_in,
                                pos_step_2d,
                                kv_c,
                                _dense_step_fn,
                                block_table=bt_kv,
                                seq_before_dev=pos_step_dev,
                                k_sconv_state=k_sc_st,
                                v_sconv_state=v_sc_st,
                                write_kv_fn=_store_decode_full_attn_kv_gpu,
                                preprojected=pre_step,
                                skip_allreduce=vrf_sp_active,
                            )
                            attn_out_steps.append(a_step)
                            if stage_kv_sc is not None and k_sc_st is not None and v_sc_st is not None:
                                stage_kv_sc[:, step, 0].copy_(k_sc_st.conv_state)
                                stage_kv_sc[:, step, 1].copy_(v_sc_st.conv_state)
                    else:
                        src_kv_c, src_bt_kv, active_kv = full_kv_layer_caches[layer.kv_cache_owner]
                        for step in range(verify_w):
                            pos_step_dev = seq_lens_before_dev + step
                            pos_step_2d = positions_2d[:, step : step + 1]
                            step_in = attn_in[:, step : step + 1]
                            visible_lens = torch.where(active_kv, pos_step_dev + 1, 0)

                            def _dense_cons_step_fn(
                                q_3d: torch.Tensor,
                                kv_l: Any,
                                _bt: torch.Tensor = src_bt_kv,
                                _vl: torch.Tensor = visible_lens,
                                _act: torch.Tensor = active_kv,
                            ) -> torch.Tensor:
                                return verify_handle.forward_dense_step(
                                    q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, active_rows=_act
                                )

                            pre_step = (
                                pre_vrf[0][:, step : step + 1],
                                None,
                                None,
                                None,
                                None,
                            )
                            a_step = sa.forward_paged_decode(
                                step_in,
                                pos_step_2d,
                                src_kv_c,
                                _dense_cons_step_fn,
                                preprojected=pre_step,
                                skip_allreduce=vrf_sp_active,
                            )
                            attn_out_steps.append(a_step)
                    attn_out = torch.cat(attn_out_steps, dim=1)

            if vrf_sp_active:
                attn_out = self._sp_reduce_scatter_dim0(
                    attn_out.reshape(bsz * verify_w, h_size),
                    total_tokens=bsz * verify_w,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                ).view(loc_bsz, loc_v, h_size)

            # 4. Attention Output ShortConv + clip (Batch-SP or Step-SP in SP domain)
            if layer.attn_output_sconv is not None:
                tag_out = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao = layer_caches[tag_out]
                bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                ao_shape = (
                    self.cfg.hidden_size,
                    layer.attn_output_sconv.kernel_size - 1,
                )
                stage_ao_full = tx_stage.stage_tensor(
                    tag_out, lid, c_ao, bt_ao, ao_shape, dtype
                )
                if vrf_sp_mode == "step":
                    attn_out = self._run_sp_step_verify_sconv(
                        attn_out,
                        layer.attn_output_sconv,
                        c_ao,
                        bt_ao,
                        seq_before_dev=seq_lens_before_dev,
                        tx_stage=tx_stage,
                        tag=tag_out,
                        lid=lid,
                        state_shape=ao_shape,
                        stage_slice=stage_ao_full[:1, v0:v1],
                        v0=v0,
                        total_v=verify_w,
                        sp_size=eff_sp,
                        sp_rank=eff_rk,
                        sp_group=eff_grp,
                        sp_comm=eff_comm,
                        clamp_val=self.cfg.clip_attn_output,
                    )
                else:
                    bt_ao_loc = bt_ao[b0:b1] if vrf_sp_mode == "batch" else bt_ao
                    stage_ao_loc = stage_ao_full[b0:b1] if vrf_sp_mode == "batch" else stage_ao_full[:bsz]
                    attn_out = self._run_verify_sconv_opaque_or_static(
                        attn_out,
                        layer.attn_output_sconv,
                        c_ao,
                        bt_ao_loc,
                        seq_loc_dev,
                        tx_stage,
                        tag_out,
                        lid,
                        ao_shape,
                        stage_ao_loc,
                        clamp_val=self.cfg.clip_attn_output,
                    )
            elif self.cfg.clip_attn_output is not None:
                attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)

            # 5. MLP Hyper-Connection + MoE/Dense MLP (1 batched pass across loc_bsz*loc_v tokens)
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                layer.mlp_hyper_connection.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
            assert mlp_inj is not None
            mlp_out = _run_runtime_mlp(
                layer,
                mlp_in,
                sp_input=vrf_sp_active,
                sp_total_tokens=(bsz * verify_w) if vrf_sp_active else None,
            )

            # 6. FFN Output ShortConv + clip (Batch-SP or Step-SP in SP domain)
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                fo_shape = (
                    self.cfg.hidden_size,
                    layer.ffn_output_sconv.kernel_size - 1,
                )
                stage_fo_full = tx_stage.stage_tensor(
                    FFN_OUTPUT_SCONV_TAG, lid, c_fo, bt_fo, fo_shape, dtype
                )
                if vrf_sp_mode == "step":
                    mlp_out = self._run_sp_step_verify_sconv(
                        mlp_out,
                        layer.ffn_output_sconv,
                        c_fo,
                        bt_fo,
                        seq_before_dev=seq_lens_before_dev,
                        tx_stage=tx_stage,
                        tag=FFN_OUTPUT_SCONV_TAG,
                        lid=lid,
                        state_shape=fo_shape,
                        stage_slice=stage_fo_full[:1, v0:v1],
                        v0=v0,
                        total_v=verify_w,
                        sp_size=eff_sp,
                        sp_rank=eff_rk,
                        sp_group=eff_grp,
                        sp_comm=eff_comm,
                        clamp_val=self.cfg.clip_mlp_output,
                    )
                else:
                    bt_fo_loc = bt_fo[b0:b1] if vrf_sp_mode == "batch" else bt_fo
                    stage_fo_loc = stage_fo_full[b0:b1] if vrf_sp_mode == "batch" else stage_fo_full[:bsz]
                    mlp_out = self._run_verify_sconv_opaque_or_static(
                        mlp_out,
                        layer.ffn_output_sconv,
                        c_fo,
                        bt_fo_loc,
                        seq_loc_dev,
                        tx_stage,
                        FFN_OUTPUT_SCONV_TAG,
                        lid,
                        fo_shape,
                        stage_fo_loc,
                        clamp_val=self.cfg.clip_mlp_output,
                    )
            elif self.cfg.clip_mlp_output is not None:
                mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)

            deferred = (mlp_out, mlp_inj)

        assert deferred is not None
        hidden, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
            deferred[0], hyper_hidden, deferred[1], prev_latent=None
        )
        tx_stage.mark_ready()
        hidden_out = hidden.reshape(loc_bsz * loc_v, h_size)
        if vrf_sp_active:
            hidden_out = self._sp_all_gather_dim0(
                hidden_out,
                total_tokens=bsz * verify_w,
                sp_size=eff_sp,
                sp_rank=eff_rk,
                sp_group=eff_grp,
                sp_comm=eff_comm,
            )
            if aux_out is not None:
                aux_out = self._sp_all_gather_dim0(
                    aux_out,
                    total_tokens=bsz * verify_w,
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    sp_group=eff_grp,
                    sp_comm=eff_comm,
                )
        if aux_out is not None:
            return hidden_out, aux_out
        return hidden_out


@dataclass
class _TxStageEntry:
    tag: str
    layer_idx: int
    cache: Any
    bt: torch.Tensor
    steps: torch.Tensor  # [B_cap, V, *shape]
    phys_ids_buf: torch.Tensor  # [2 * B_cap] int32
    commit_u8_buf: torch.Tensor  # [2 * B_cap, need_bytes] uint8
    undo_u8_buf: torch.Tensor  # [2 * B_cap, need_bytes] uint8
    need_bytes: int
    gen: int = 0


@dataclass
class _LazyGdnEntry:
    tag: str
    layer_idx: int
    cache: Any
    bt: torch.Tensor
    init_ssm: torch.Tensor  # [B_cap, ssm_n] float32
    lazy_k: torch.Tensor  # [B_cap, V, H_k, D_k]
    lazy_v: torch.Tensor  # [B_cap, V, H_v, D_v]
    lazy_g: torch.Tensor  # [B_cap, V, H_v] float32
    lazy_beta: torch.Tensor  # [B_cap, V, H_v] float32
    conv_steps: Optional[torch.Tensor]  # [B_cap, V, *conv_shape] bf16
    phys_ids_buf: torch.Tensor  # [2 * B_cap] int32
    conv_phys_buf: torch.Tensor  # [2 * B_cap] int32
    commit_u8_buf: torch.Tensor  # [2 * B_cap, (ssm_n + conv_i32_elems) * 4] uint8
    undo_u8_buf: torch.Tensor  # [2 * B_cap, (ssm_n + conv_i32_elems) * 4] uint8
    ssm_n: int
    conv_i32_elems: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    gen: int = 0


@dataclass
class _TxUndoRecord:
    cache: Any
    tag: str
    layer_idx: int
    phys_ids: torch.Tensor
    commit_u8: torch.Tensor
    undo_u8: torch.Tensor


class _Qwen4_5OpaqueTxStage:
    """Static-address staging & commit/undo buffers for a fixed ``(B_cap, V)`` verify shape."""

    def __init__(self, bsz: int, verify_width: int) -> None:
        self.bsz = int(bsz)
        self.verify_width = int(verify_width)
        self.entries: Dict[Tuple[str, int], _TxStageEntry] = {}
        self.lazy_gdn_entries: Dict[Tuple[str, int], _LazyGdnEntry] = {}
        self.scratches: Dict[Any, torch.Tensor] = {}
        self._scratch_bts: Dict[int, torch.Tensor] = {}
        self._steps_arange: Optional[torch.Tensor] = None
        self.seq_before_buf: Optional[torch.Tensor] = None
        self.accept_len_buf: Optional[torch.Tensor] = None
        self.aux_buf: Optional[torch.Tensor] = None
        self.attention_inputs: Any = None
        self.kv_cache: Any = None
        self.ready = False
        self._gen = 0
        self._batched_sig: Any = None
        self._batched_meta: Optional[Dict[str, Any]] = None
        self._lazy_sig: Any = None
        self._lazy_meta: Optional[List[Dict[str, Any]]] = None
        self._prepared_plan: Any = None
        self.sp_mode: Optional[str] = None
        self.sp_bounds: Optional[Tuple[int, int]] = None

    def get_steps_arange(self, verify_w: int, device: torch.device) -> torch.Tensor:
        if (
            self._steps_arange is None
            or int(self._steps_arange.numel()) != verify_w
            or self._steps_arange.device != device
        ):
            self._steps_arange = torch.arange(verify_w, dtype=torch.int32, device=device).unsqueeze(0)
        return self._steps_arange

    def begin_forward(
        self,
        seq_before_dev: torch.Tensor,
        kv_cache: Any,
        attention_inputs: Any,
    ) -> None:
        if (
            self.seq_before_buf is None
            or self.seq_before_buf.device != seq_before_dev.device
            or self.seq_before_buf.numel() != self.bsz
        ):
            self.seq_before_buf = torch.empty(
                (self.bsz,), dtype=torch.int32, device=seq_before_dev.device
            )
            self.accept_len_buf = torch.empty(
                (self.bsz,), dtype=torch.int32, device=seq_before_dev.device
            )
        self.seq_before_buf.copy_(seq_before_dev[: self.bsz].to(torch.int32))
        self.kv_cache = kv_cache
        self.attention_inputs = attention_inputs
        self.ready = False
        self._prepared_plan = None
        self._gen += 1

    def get_or_alloc_aux(
        self, rows: int, cols: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if (
            self.aux_buf is None
            or tuple(self.aux_buf.shape) != (rows, cols)
            or self.aux_buf.dtype != dtype
            or self.aux_buf.device != device
        ):
            self.aux_buf = torch.empty((rows, cols), dtype=dtype, device=device)
        return self.aux_buf

    def stage_tensor(
        self,
        tag: str,
        layer_idx: int,
        cache: Any,
        bt: torch.Tensor,
        state_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (tag, int(layer_idx))
        wanted_shape = (self.bsz, self.verify_width, *state_shape)
        elem_bytes = _dtype_itemsize(dtype)
        need_bytes = math.prod(state_shape) * elem_bytes
        entry = self.entries.get(key)
        if (
            entry is None
            or tuple(entry.steps.shape) != wanted_shape
            or entry.steps.dtype != dtype
            or entry.steps.device != bt.device
        ):
            buf = torch.empty(wanted_shape, dtype=dtype, device=bt.device)
            phys_ids_buf = torch.empty((2 * self.bsz,), dtype=torch.int32, device=bt.device)
            commit_u8_buf = torch.empty((2 * self.bsz, need_bytes), dtype=torch.uint8, device=bt.device)
            undo_u8_buf = torch.empty((2 * self.bsz, need_bytes), dtype=torch.uint8, device=bt.device)
            entry = _TxStageEntry(
                tag=tag,
                layer_idx=int(layer_idx),
                cache=cache,
                bt=bt,
                steps=buf,
                phys_ids_buf=phys_ids_buf,
                commit_u8_buf=commit_u8_buf,
                undo_u8_buf=undo_u8_buf,
                need_bytes=need_bytes,
                gen=self._gen,
            )
            self.entries[key] = entry
        else:
            entry.cache = cache
            entry.bt = bt
            entry.gen = self._gen
        return entry.steps

    def register_lazy_gdn(
        self,
        tag: str,
        layer_idx: int,
        cache: Any,
        bt: torch.Tensor,
        init_ssm: torch.Tensor,
        *,
        ssm_n: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        kv_dtype: torch.dtype,
        needs_conv_stage: bool = False,
        conv_shape: Optional[Tuple[int, ...]] = None,
        conv_i32_elems: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        key = (tag, int(layer_idx))
        dev = bt.device
        k_shape = (self.bsz, self.verify_width, num_k_heads, head_k_dim)
        v_shape = (self.bsz, self.verify_width, num_v_heads, head_v_dim)
        gb_shape = (self.bsz, self.verify_width, num_v_heads)
        need_bytes = (int(ssm_n) + (int(conv_i32_elems) if needs_conv_stage else 0)) * 4
        conv_st_shape = (self.bsz, self.verify_width, *conv_shape) if (needs_conv_stage and conv_shape is not None) else None
        entry = self.lazy_gdn_entries.get(key)
        if (
            entry is None
            or tuple(entry.lazy_k.shape) != k_shape
            or entry.lazy_k.dtype != kv_dtype
            or entry.lazy_k.device != dev
            or int(entry.commit_u8_buf.shape[1]) != need_bytes
            or (entry.conv_steps is None) != (conv_st_shape is None)
            or (conv_st_shape is not None and entry.conv_steps is not None and tuple(entry.conv_steps.shape) != conv_st_shape)
        ):
            lazy_k = self.scratch_tensor(("lazy_k", tag, int(layer_idx)), k_shape, kv_dtype, dev)
            lazy_v = self.scratch_tensor(("lazy_v", tag, int(layer_idx)), v_shape, kv_dtype, dev)
            lazy_g = self.scratch_tensor(("lazy_g", tag, int(layer_idx)), gb_shape, torch.float32, dev)
            lazy_beta = self.scratch_tensor(("lazy_beta", tag, int(layer_idx)), gb_shape, torch.float32, dev)
            conv_steps = (
                self.scratch_tensor(("lazy_conv", tag, int(layer_idx)), conv_st_shape, kv_dtype, dev)
                if conv_st_shape is not None
                else None
            )
            phys_ids_buf = torch.empty((2 * self.bsz,), dtype=torch.int32, device=dev)
            conv_phys_buf = torch.empty((2 * self.bsz,), dtype=torch.int32, device=dev)
            commit_u8_buf = torch.empty((2 * self.bsz, need_bytes), dtype=torch.uint8, device=dev)
            undo_u8_buf = torch.empty((2 * self.bsz, need_bytes), dtype=torch.uint8, device=dev)
            entry = _LazyGdnEntry(
                tag=tag,
                layer_idx=int(layer_idx),
                cache=cache,
                bt=bt,
                init_ssm=init_ssm,
                lazy_k=lazy_k,
                lazy_v=lazy_v,
                lazy_g=lazy_g,
                lazy_beta=lazy_beta,
                conv_steps=conv_steps,
                phys_ids_buf=phys_ids_buf,
                conv_phys_buf=conv_phys_buf,
                commit_u8_buf=commit_u8_buf,
                undo_u8_buf=undo_u8_buf,
                ssm_n=int(ssm_n),
                conv_i32_elems=int(conv_i32_elems) if needs_conv_stage else 0,
                num_k_heads=int(num_k_heads),
                num_v_heads=int(num_v_heads),
                head_k_dim=int(head_k_dim),
                head_v_dim=int(head_v_dim),
                gen=self._gen,
            )
            self.lazy_gdn_entries[key] = entry
        else:
            entry.cache = cache
            entry.bt = bt
            entry.init_ssm = init_ssm
            entry.gen = self._gen
        return entry.lazy_k, entry.lazy_v, entry.lazy_g, entry.lazy_beta, entry.conv_steps

    def scratch_tensor(
        self,
        key: Any,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buf = self.scratches.get(key)
        if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype or buf.device != device:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self.scratches[key] = buf
        return buf

    def get_idx_scratch_bt(
        self,
        layer_idx: int,
        bsz: int,
        max_cols: int,
        active_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        buf = self._scratch_bts.get(layer_idx)
        if buf is None or tuple(buf.shape) != (bsz, max_cols) or buf.device != device:
            buf = torch.empty((bsz, max_cols), dtype=torch.int32, device=device)
            self._scratch_bts[layer_idx] = buf
        row_ids = torch.where(
            active_idx,
            torch.arange(1, bsz + 1, dtype=torch.int32, device=device),
            0,
        )
        buf.copy_(row_ids.unsqueeze(1).expand(bsz, max_cols))
        return buf

    def mark_ready(self) -> None:
        self.ready = True


def _build_batched_tx_meta(specs: List[Tuple[Any, ...]], block_sz: int, dev: torch.device, *, with_steps: bool) -> Dict[str, Any]:
    tile_e_list: List[int] = []
    tile_blk_list: List[int] = []
    for ei, s in enumerate(specs):
        nt = int(triton.cdiv(s[17], block_sz))
        tile_e_list.extend([ei] * nt)
        tile_blk_list.extend(range(nt))
    i32_fields: List[Tuple[str, List[int]]] = [
        ("tile_e", tile_e_list),
        ("tile_blk", tile_blk_list),
        ("num_blocks", [s[16] for s in specs]),
        ("n_elems", [s[17] for s in specs]),
    ]
    i64_fields: List[Tuple[str, List[int]]] = [
        ("ptrs_base", [s[3].data_ptr() for s in specs]),
        ("ptrs_phys", [s[4].data_ptr() for s in specs]),
        ("ptrs_commit", [s[5].data_ptr() for s in specs]),
        ("ptrs_undo", [s[6].data_ptr() for s in specs]),
        ("base_sb", [s[11] for s in specs]),
        ("commit_sb", [s[12] for s in specs]),
        ("undo_sb", [s[13] for s in specs]),
    ]
    if with_steps:
        i32_fields.extend(
            [
                ("tpb", [s[14] for s in specs]),
                ("max_blocks", [s[15] for s in specs]),
            ]
        )
        i64_fields.extend(
            [
                ("ptrs_bt", [s[1].data_ptr() for s in specs]),
                ("ptrs_steps", [s[2].data_ptr() for s in specs]),
                ("bt_s0", [s[7] for s in specs]),
                ("bt_s1", [s[8] for s in specs]),
                ("steps_sb", [s[9] for s in specs]),
                ("steps_sv", [s[10] for s in specs]),
            ]
        )
    flat_32: List[int] = []
    spans_32: List[Tuple[str, int, int]] = []
    for nm, lst in i32_fields:
        a = len(flat_32)
        flat_32.extend(lst)
        spans_32.append((nm, a, a + len(lst)))
        if (len(lst) & 1) != 0:
            flat_32.append(0)
    flat_64: List[int] = []
    spans_64: List[Tuple[str, int, int]] = []
    for nm, lst in i64_fields:
        a = len(flat_64)
        flat_64.extend(lst)
        spans_64.append((nm, a, a + len(lst)))
    dev_32 = _copy_i32_list_to_dev(flat_32, dev)
    arr_64 = array.array("q", flat_64)
    dev_64 = _pin_stage_copy_to_device(torch.frombuffer(arr_64, dtype=torch.int64), dev)
    meta: Dict[str, Any] = {"num_e": len(specs), "num_tiles": len(tile_e_list)}
    for nm, a, b in spans_32:
        meta[nm] = dev_32[a:b]
    for nm, a, b in spans_64:
        meta[nm] = dev_64[a:b]
    return meta


_SP_VERIFY_SHARDED_TAGS = frozenset(
    {
        PLE_STATE_TAG,
        PLE_NGRAM_CTX_TAG,
        FULL_ATTN_OUTPUT_SCONV_TAG,
        LINEAR_ATTN_OUTPUT_SCONV_TAG,
        FFN_OUTPUT_SCONV_TAG,
    }
)


class _Qwen4_5OpaqueTxManager:
    """Two-phase commit + rollback manager for speculative ``OPAQUE_STATE`` pools (zero dynamic allocs)."""

    def _sp_sync_step_before_tx(
        self,
        stage: _Qwen4_5OpaqueTxStage,
        accept_len: torch.Tensor,
        live_bs: int,
        v_w: int,
        dev: torch.device,
    ) -> None:
        if stage.sp_mode != "step" or live_bs != 1:
            return
        eff_sp, eff_rk, eff_grp, _ = self.bridge._resolve_sp_dist()
        if eff_sp <= 1 or eff_grp is None:
            return
        assert stage.seq_before_buf is not None
        if accept_len.device == dev:
            acc0, pos_b = [
                int(x)
                for x in torch.stack(
                    [accept_len[0].to(torch.int64), stage.seq_before_buf[0].to(torch.int64)]
                ).tolist()
            ]
        else:
            acc0 = int(accept_len[0].item())
            pos_b = int(stage.seq_before_buf[0].item())
        if acc0 <= 0:
            return
        pos_b = max(0, pos_b)
        entries = sorted(
            (
                e
                for e in stage.entries.values()
                if e.gen == stage._gen and e.tag in _SP_VERIFY_SHARDED_TAGS
            ),
            key=lambda e: (e.tag, e.layer_idx),
        )
        if not entries:
            return
        end_step = min(v_w - 1, acc0 - 1)
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        bounds = [sp_slice_bounds(v_w, eff_sp, r_i) for r_i in range(eff_sp)]

        def _owner_for_step(s_idx: int) -> int:
            for r_i, (r0, r1) in enumerate(bounds):
                if r0 <= s_idx < r1:
                    return r_i
            return 0

        items_by_owner: Dict[int, List[Tuple[_TxStageEntry, int]]] = {}
        for e in entries:
            tpb = max(1, int(e.cache.seq_size_per_block))
            needed_steps = [end_step]
            boundary_step = (tpb - 1) - (pos_b % tpb)
            if (acc0 - 1) > boundary_step and 0 <= boundary_step < v_w and boundary_step != end_step:
                needed_steps.append(boundary_step)
            for s_idx in needed_steps:
                items_by_owner.setdefault(_owner_for_step(s_idx), []).append((e, s_idx))

        for owner_rk, pairs in sorted(items_by_owner.items()):
            total_b = sum(
                e.steps[0, s_idx].numel() * e.steps.element_size() for e, s_idx in pairs
            )
            if total_b <= 0:
                continue
            buf = torch.empty((total_b,), dtype=torch.uint8, device=dev)
            if eff_rk == owner_rk:
                off = 0
                for e, s_idx in pairs:
                    s_u8 = e.steps[0, s_idx].contiguous().view(torch.uint8).reshape(-1)
                    nb = int(s_u8.numel())
                    buf[off : off + nb].copy_(s_u8)
                    off += nb
            src_global = dist.get_global_rank(eff_grp, owner_rk)
            dist.broadcast(buf, src=src_global, group=eff_grp)
            if eff_rk != owner_rk:
                off = 0
                for e, s_idx in pairs:
                    dst = e.steps[0, s_idx]
                    nb = int(dst.numel() * dst.element_size())
                    dst.view(torch.uint8).reshape(-1).copy_(buf[off : off + nb])
                    off += nb

    def _get_or_build_sp_batch_commit_meta(
        self,
        sp_recs: Sequence[_TxUndoRecord],
        ranges: Sequence[Tuple[int, int]],
        aligned_b: int,
        eff_sp: int,
        eff_rk: int,
        dev: torch.device,
    ) -> Optional[Dict[str, Any]]:
        if not sp_recs or not _HAS_TRITON or dev.type != "cuda":
            return None
        key = (
            dev.index if dev.index is not None else -1,
            int(eff_sp),
            int(eff_rk),
            tuple(ranges),
            tuple((r.commit_u8.data_ptr(), int(r.commit_u8.shape[1]), int(r.commit_u8.stride(0))) for r in sp_recs),
        )
        hit = self._sp_batch_commit_meta_cache.get(key)
        if hit is not None:
            self._sp_batch_commit_meta_cache.move_to_end(key)
            return hit
        if torch.cuda.is_current_stream_capturing():
            return None

        block_sz = 1024
        ptrs_c_h: List[int] = []
        c_sb_h: List[int] = []
        n_elems_h: List[int] = []
        unit_off_h: List[int] = []
        tile_e_h: List[int] = []
        tile_blk_h: List[int] = []
        c_refs: List[torch.Tensor] = []
        u_cur = 0
        for e_i, rec in enumerate(sp_recs):
            cu8 = rec.commit_u8
            nb = int(cu8.shape[1])
            sb_u8 = int(cu8.stride(0))
            if (nb % 4) != 0 or (sb_u8 % 4) != 0 or (cu8.data_ptr() % 4) != 0:
                return None
            n_elems = nb // 4
            ptrs_c_h.append(cu8.data_ptr())
            c_sb_h.append(sb_u8 // 4)
            n_elems_h.append(n_elems)
            unit_off_h.append(u_cur)
            u_cur += 2 * n_elems
            n_blks = int(triton.cdiv(n_elems, block_sz))
            for blk in range(n_blks):
                tile_e_h.append(e_i)
                tile_blk_h.append(blk)
            c_refs.append(cu8)

        if not tile_e_h:
            return None
        r_lo_h = [lo_r for lo_r, _ in ranges]
        r_hi_h = [hi_r for _, hi_r in ranges]
        segs_32 = [tile_e_h, tile_blk_h, n_elems_h, unit_off_h, r_lo_h, r_hi_h]
        flat_32: List[int] = []
        spans_32: List[Tuple[int, int]] = []
        for lst in segs_32:
            s0_l = len(flat_32)
            flat_32.extend(lst)
            pad = (-len(lst)) & 3
            if pad:
                flat_32.extend([0] * pad)
            spans_32.append((s0_l, s0_l + len(lst)))
        dev_32 = _copy_i32_list_to_dev(flat_32, dev)
        v32 = [dev_32[a:b] for (a, b) in spans_32]

        ne = len(sp_recs)
        arr_64 = array.array("q", ptrs_c_h + c_sb_h)
        dev_64 = _pin_stage_copy_to_device(torch.frombuffer(arr_64, dtype=torch.int64), dev)
        aligned_i32 = aligned_b // 4
        max_remote_cnt = max(
            (hi_r - lo_r for r_i, (lo_r, hi_r) in enumerate(ranges) if r_i != eff_rk and hi_r > lo_r),
            default=0,
        )
        ptr_or = (
            int(v32[0].data_ptr())
            | int(v32[1].data_ptr())
            | int(v32[2].data_ptr())
            | int(v32[3].data_ptr())
            | int(v32[4].data_ptr())
            | int(v32[5].data_ptr())
            | int(dev_64.data_ptr())
        )
        meta = {
            "num_e": ne,
            "num_tiles": len(tile_e_h),
            "block_sz": block_sz,
            "aligned_i32": aligned_i32,
            "max_remote_cnt": max_remote_cnt,
            "tile_e": v32[0],
            "tile_blk": v32[1],
            "n_elems": v32[2],
            "unit_off": v32[3],
            "r_lo": v32[4],
            "r_hi": v32[5],
            "ptrs_commit": dev_64[0:ne],
            "commit_sb": dev_64[ne : 2 * ne],
            "ptr_or": ptr_or,
            "_refs": tuple(c_refs) + (dev_32, dev_64),
        }
        self._sp_batch_commit_meta_cache[key] = meta
        if len(self._sp_batch_commit_meta_cache) > 64:
            self._sp_batch_commit_meta_cache.popitem(last=False)
        return meta

    def _sp_sync_batch_commit_after_tx(
        self,
        stage: _Qwen4_5OpaqueTxStage,
        undo_records: Sequence[_TxUndoRecord],
        live_bs: int,
        dev: torch.device,
    ) -> None:
        if stage.sp_mode != "batch" or live_bs <= 0:
            return
        eff_sp, eff_rk, eff_grp, eff_comm = self.bridge._resolve_sp_dist()
        if eff_sp <= 1 or eff_grp is None:
            return
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds

        sp_recs = [r for r in undo_records if r.tag in _SP_VERIFY_SHARDED_TAGS]
        if not sp_recs:
            return
        bsz_full = stage.bsz
        ranges: List[Tuple[int, int]] = []
        max_b = 0
        for r_i in range(eff_sp):
            r0, r1 = sp_slice_bounds(bsz_full, eff_sp, r_i)
            lo = min(r0, live_bs)
            hi = min(r1, live_bs)
            ranges.append((lo, hi))
            rb = sum((hi - lo) * 2 * int(r.commit_u8.shape[1]) for r in sp_recs) if hi > lo else 0
            if rb > max_b:
                max_b = rb
        if max_b <= 0:
            return
        aligned_b = ((max_b + 15) // 16) * 16
        lo_m, hi_m = ranges[eff_rk]
        meta = self._get_or_build_sp_batch_commit_meta(
            sp_recs, ranges, aligned_b, eff_sp, eff_rk, dev
        )
        if meta is not None:
            aligned_i32 = int(meta["aligned_i32"])
            num_tiles = int(meta["num_tiles"])
            send_buf = torch.empty((aligned_i32,), dtype=torch.int32, device=dev)
            cnt_m = max(0, hi_m - lo_m)
            if cnt_m > 0:
                ckey_p = (
                    dev.index,
                    "tx_sp_pack",
                    meta["num_e"],
                    _triton_int_spec(num_tiles),
                    _triton_int_spec(cnt_m),
                    meta["block_sz"],
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE,
                    ckey_p,
                    _tx_sp_batch_pack_triton,
                    num_tiles,
                    cnt_m,
                    2,
                    int(meta["ptr_or"]) | send_buf.data_ptr(),
                    (
                        meta["tile_e"],
                        meta["tile_blk"],
                        meta["ptrs_commit"],
                        meta["commit_sb"],
                        meta["n_elems"],
                        meta["unit_off"],
                        send_buf,
                        lo_m,
                        cnt_m,
                        live_bs,
                    ),
                    {"BLOCK_SIZE": meta["block_sz"]},
                    num_warps=4,
                )
            all_bf16 = self.bridge._sp_all_gather_dim0(
                send_buf.view(torch.bfloat16).unsqueeze(0),
                total_tokens=eff_sp,
                sp_size=eff_sp,
                sp_rank=eff_rk,
                sp_group=eff_grp,
                sp_comm=eff_comm,
            )
            max_remote_cnt = int(meta["max_remote_cnt"])
            if max_remote_cnt > 0:
                all_buf = all_bf16.contiguous().view(torch.int32).reshape(eff_sp, aligned_i32)
                all_s0 = int(all_buf.stride(0))
                ckey_u = (
                    dev.index,
                    "tx_sp_unpk",
                    meta["num_e"],
                    _triton_int_spec(num_tiles),
                    _triton_int_spec(max_remote_cnt),
                    eff_sp,
                    _triton_int_spec(all_s0),
                    meta["block_sz"],
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE,
                    ckey_u,
                    _tx_sp_batch_unpack_triton,
                    num_tiles,
                    max_remote_cnt,
                    eff_sp * 2,
                    int(meta["ptr_or"]) | all_buf.data_ptr(),
                    (
                        meta["tile_e"],
                        meta["tile_blk"],
                        meta["ptrs_commit"],
                        meta["commit_sb"],
                        meta["n_elems"],
                        meta["unit_off"],
                        meta["r_lo"],
                        meta["r_hi"],
                        all_buf,
                        all_s0,
                        live_bs,
                        eff_rk,
                    ),
                    {"BLOCK_SIZE": meta["block_sz"]},
                    num_warps=4,
                )
            return

        send_buf = torch.zeros((aligned_b,), dtype=torch.uint8, device=dev)
        if hi_m > lo_m:
            off = 0
            for rec in sp_recs:
                nb = int(rec.commit_u8.shape[1])
                cnt = hi_m - lo_m
                s1 = rec.commit_u8[lo_m:hi_m].reshape(-1)
                s2 = rec.commit_u8[live_bs + lo_m : live_bs + hi_m].reshape(-1)
                send_buf[off : off + cnt * nb].copy_(s1)
                send_buf[off + cnt * nb : off + 2 * cnt * nb].copy_(s2)
                off += 2 * cnt * nb
        all_buf = self.bridge._sp_all_gather_dim0(
            send_buf,
            total_tokens=eff_sp * aligned_b,
            sp_size=eff_sp,
            sp_rank=eff_rk,
            sp_group=eff_grp,
            sp_comm=eff_comm,
        ).view(eff_sp, aligned_b)
        for r_i, (lo_r, hi_r) in enumerate(ranges):
            if r_i == eff_rk or hi_r <= lo_r:
                continue
            off = 0
            cnt = hi_r - lo_r
            rb = all_buf[r_i]
            for rec in sp_recs:
                nb = int(rec.commit_u8.shape[1])
                rec.commit_u8[lo_r:hi_r].copy_(rb[off : off + cnt * nb].view(cnt, nb))
                rec.commit_u8[live_bs + lo_r : live_bs + hi_r].copy_(rb[off + cnt * nb : off + 2 * cnt * nb].view(cnt, nb))
                off += 2 * cnt * nb

    def __init__(self, bridge: Qwen4_5PagedStateBridge, arch_cfg: Qwen4_5ArchConfig) -> None:
        self.bridge = bridge
        self.arch_cfg = arch_cfg
        self._graph_stages: Dict[Tuple[int, int], _Qwen4_5OpaqueTxStage] = {}
        self._eager_stage: Optional[_Qwen4_5OpaqueTxStage] = None
        self._active_pending: Optional[_Qwen4_5OpaqueTxStage] = None
        self._undo_records: List[_TxUndoRecord] = []
        self._committed = False
        self._batched_undo_meta: Optional[Dict[str, Any]] = None
        self._sp_batch_commit_meta_cache: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()

    def acquire_stage(
        self,
        bsz: int,
        verify_width: int,
        *,
        is_cuda_graph: bool,
    ) -> _Qwen4_5OpaqueTxStage:
        if is_cuda_graph:
            key = (int(bsz), int(verify_width))
            stage = self._graph_stages.get(key)
            if stage is None:
                stage = _Qwen4_5OpaqueTxStage(bsz, verify_width)
                self._graph_stages[key] = stage
            self._active_pending = stage
            return stage
        if (
            self._eager_stage is None
            or self._eager_stage.bsz != bsz
            or self._eager_stage.verify_width != verify_width
        ):
            self._eager_stage = _Qwen4_5OpaqueTxStage(bsz, verify_width)
        self._active_pending = self._eager_stage
        return self._eager_stage

    def on_cuda_graph_warmup(self) -> None:
        self._active_pending = None

    def on_cuda_graph_capture(self) -> None:
        self._active_pending = None

    def on_cuda_graph_replay(
        self,
        graph_key: int,
        verify_width: int,
        seq_before_dev: torch.Tensor,
        attention_inputs: Any,
    ) -> None:
        key = (int(graph_key), int(verify_width))
        stage = self._graph_stages.get(key)
        if stage is None:
            raise RuntimeError(f"Missing CUDA graph OPAQUE_STATE stage for {key}")
        if stage.seq_before_buf is not None:
            stage.seq_before_buf.copy_(seq_before_dev[: stage.bsz].to(torch.int32))
        stage.attention_inputs = attention_inputs
        for entry in stage.entries.values():
            if entry.gen == stage._gen:
                entry.bt = _block_table_for_tag(attention_inputs, entry.tag)
        for lentry in stage.lazy_gdn_entries.values():
            if lentry.gen == stage._gen:
                lentry.bt = _block_table_for_tag(attention_inputs, lentry.tag)
        stage.ready = True
        self._active_pending = stage

    @staticmethod
    def _launch_tx_kernels(
        stage: _Qwen4_5OpaqueTxStage,
        live_bs: int,
        v_w: int,
        lazy_meta_groups: Optional[List[Dict[str, Any]]],
        batched_meta: Optional[Dict[str, Any]],
        prep_specs: List[Tuple[Any, ...]],
    ) -> None:
        assert stage.seq_before_buf is not None and stage.accept_len_buf is not None
        if lazy_meta_groups:
            for lm in lazy_meta_groups:
                hk, hv, dk, dv, sk_dt = lm["shape_key"]
                bdv = min(16, dv)
                grid_x = lm["num_e"] * hv * (dv // bdv)
                rt_args = (
                    lm["sample_kv"],
                    lm["ptrs_bt"], lm["ptrs_init"], lm["ptrs_k"], lm["ptrs_v"],
                    lm["ptrs_g"], lm["ptrs_beta"], lm["ptrs_base"], lm["ptrs_phys"],
                    lm["ptrs_commit"], lm["ptrs_undo"],
                    stage.seq_before_buf, stage.accept_len_buf,
                    lm["bt_s0"], lm["bt_s1"], lm["init_sb"],
                    lm["k_sb"], lm["k_sv"], lm["k_shk"], lm["k_sd"],
                    lm["v_sb"], lm["v_sv"], lm["v_shv"], lm["v_vd"],
                    lm["g_sb"], lm["g_sv"], lm["g_shv"],
                    lm["b_sb"], lm["b_sv"], lm["b_shv"],
                    lm["base_sb"], lm["commit_sb"], lm["undo_sb"],
                    lm["tpb"], lm["max_blocks"], lm["num_blocks"],
                    live_bs,
                )
                ptr_or = (
                    lm["sample_kv"].data_ptr()
                    | lm["ptrs_bt"].data_ptr() | lm["ptrs_init"].data_ptr()
                    | lm["ptrs_k"].data_ptr() | lm["ptrs_v"].data_ptr()
                    | lm["ptrs_g"].data_ptr() | lm["ptrs_beta"].data_ptr()
                    | lm["ptrs_base"].data_ptr() | lm["ptrs_phys"].data_ptr()
                    | lm["ptrs_commit"].data_ptr() | lm["ptrs_undo"].data_ptr()
                    | stage.seq_before_buf.data_ptr() | stage.accept_len_buf.data_ptr()
                    | lm["bt_s0"].data_ptr() | lm["bt_s1"].data_ptr() | lm["init_sb"].data_ptr()
                    | lm["k_sb"].data_ptr() | lm["k_sv"].data_ptr() | lm["k_shk"].data_ptr() | lm["k_sd"].data_ptr()
                    | lm["v_sb"].data_ptr() | lm["v_sv"].data_ptr() | lm["v_shv"].data_ptr() | lm["v_vd"].data_ptr()
                    | lm["g_sb"].data_ptr() | lm["g_sv"].data_ptr() | lm["g_shv"].data_ptr()
                    | lm["b_sb"].data_ptr() | lm["b_sv"].data_ptr() | lm["b_shv"].data_ptr()
                    | lm["base_sb"].data_ptr() | lm["commit_sb"].data_ptr() | lm["undo_sb"].data_ptr()
                    | lm["tpb"].data_ptr() | lm["max_blocks"].data_ptr() | lm["num_blocks"].data_ptr()
                )
                ckey = (
                    stage.seq_before_buf.device.index, "tx_lazy_gdn",
                    sk_dt, stage.seq_before_buf.dtype, stage.accept_len_buf.dtype,
                    _triton_int_spec(live_bs), v_w, hv // hk, hv, dk, dv, bdv,
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE, ckey, _prepare_gdn_ssm_lazy_commit_batched_triton,
                    grid_x, 2 * live_bs, 1, ptr_or, rt_args,
                    {
                        "V_STEPS": v_w, "GQA_RATIO": hv // hk, "H_V": hv,
                        "D_K": dk, "D_V": dv, "BLOCK_DV": bdv, "L2_EPS": 1e-6,
                    },
                    num_warps=2,
                )
        if batched_meta is not None:
            pm = batched_meta["prep"]
            if pm is not None and pm["num_tiles"] > 0:
                bsz_blk = int(batched_meta["block_sz"])
                rt_args = (
                    pm["tile_e"], pm["tile_blk"],
                    pm["ptrs_bt"], pm["ptrs_steps"], pm["ptrs_base"],
                    pm["ptrs_phys"], pm["ptrs_commit"], pm["ptrs_undo"],
                    stage.seq_before_buf, stage.accept_len_buf,
                    pm["bt_s0"], pm["bt_s1"], pm["steps_sb"], pm["steps_sv"],
                    pm["base_sb"], pm["commit_sb"], pm["undo_sb"],
                    pm["tpb"], pm["max_blocks"], pm["num_blocks"], pm["n_elems"],
                    live_bs, v_w,
                )
                ptr_or = (
                    pm["tile_e"].data_ptr() | pm["tile_blk"].data_ptr()
                    | pm["ptrs_bt"].data_ptr() | pm["ptrs_steps"].data_ptr() | pm["ptrs_base"].data_ptr()
                    | pm["ptrs_phys"].data_ptr() | pm["ptrs_commit"].data_ptr() | pm["ptrs_undo"].data_ptr()
                    | stage.seq_before_buf.data_ptr() | stage.accept_len_buf.data_ptr()
                    | pm["bt_s0"].data_ptr() | pm["bt_s1"].data_ptr() | pm["steps_sb"].data_ptr() | pm["steps_sv"].data_ptr()
                    | pm["base_sb"].data_ptr() | pm["commit_sb"].data_ptr() | pm["undo_sb"].data_ptr()
                    | pm["tpb"].data_ptr() | pm["max_blocks"].data_ptr() | pm["num_blocks"].data_ptr() | pm["n_elems"].data_ptr()
                )
                ckey = (
                    stage.seq_before_buf.device.index, "tx_prep_batch",
                    stage.seq_before_buf.dtype, stage.accept_len_buf.dtype,
                    _triton_int_spec(live_bs), _triton_int_spec(v_w), bsz_blk,
                )
                triton_fast_launch(
                    _RUNTIME_TRITON_CACHE, ckey, _prepare_opaque_tx_commit_batched_triton,
                    pm["num_tiles"], 2 * live_bs, 1, ptr_or, rt_args,
                    {"BLOCK_SIZE": bsz_blk},
                    num_warps=4,
                )
        elif prep_specs:
            bsz_blk = 2048
            for s in prep_specs:
                _prepare_opaque_tx_commit_triton[(2 * live_bs, int(triton.cdiv(s[17], bsz_blk)))](
                    s[1], stage.seq_before_buf, stage.accept_len_buf,
                    s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13],
                    live_bs, v_w, s[14], s[15], s[16], s[17],
                    BLOCK_SIZE=bsz_blk, num_warps=4,
                )

    @staticmethod
    def _launch_write_batched(wm: Dict[str, Any], src_key: str, lbs: int, block_sz: int) -> None:
        rt_args = (
            wm["tile_e"], wm["tile_blk"], wm["ptrs_base"], wm[src_key], wm["ptrs_phys"],
            wm["base_sb"], wm["commit_sb"] if src_key == "ptrs_commit" else wm["undo_sb"],
            wm["num_blocks"], wm["n_elems"],
        )
        sb_src = wm["commit_sb"] if src_key == "ptrs_commit" else wm["undo_sb"]
        ptr_or = (
            wm["tile_e"].data_ptr() | wm["tile_blk"].data_ptr() | wm["ptrs_base"].data_ptr()
            | wm[src_key].data_ptr() | wm["ptrs_phys"].data_ptr() | wm["base_sb"].data_ptr()
            | sb_src.data_ptr() | wm["num_blocks"].data_ptr() | wm["n_elems"].data_ptr()
        )
        ckey = (wm["tile_e"].device.index, "tx_write_batch", block_sz)
        triton_fast_launch(
            _RUNTIME_TRITON_CACHE, ckey, _write_opaque_rows_batched_triton,
            wm["num_tiles"], 2 * lbs, 1, ptr_or, rt_args,
            {"BLOCK_SIZE": block_sz},
            num_warps=4,
        )

    def prepare_commit(self, accept_len: torch.Tensor) -> None:
        if self._undo_records:
            raise RuntimeError("Concurrent uncommitted speculative target transaction")
        stage = self._active_pending
        if stage is None or not stage.ready or stage.seq_before_buf is None:
            return

        live_bs = int(accept_len.shape[0])
        if live_bs <= 0:
            return
        if live_bs > stage.bsz:
            raise ValueError(f"accept_len batch {live_bs} exceeds staged batch {stage.bsz}")

        dev = stage.seq_before_buf.device
        assert stage.accept_len_buf is not None
        if accept_len.device == dev:
            stage.accept_len_buf[:live_bs].copy_(accept_len.to(dtype=torch.int32))
        else:
            acc_pin = _pin_stage_copy_to_device(accept_len.contiguous().to(dtype=torch.int32).reshape(-1), dev)
            stage.accept_len_buf[:live_bs].copy_(acc_pin)
        v_w = stage.verify_width
        self._sp_sync_step_before_tx(stage, accept_len, live_bs, v_w, dev)

        bt_key = (
            stage._gen,
            live_bs,
            tuple(
                (e.bt.data_ptr(), int(e.bt.shape[1]) if e.bt.ndim >= 2 else 0, int(e.bt.stride(0)), int(e.bt.stride(1)) if e.bt.ndim >= 2 else 0)
                for e in stage.entries.values()
                if e.gen == stage._gen
            ),
            tuple(
                (l.bt.data_ptr(), int(l.bt.shape[1]) if l.bt.ndim >= 2 else 0, int(l.bt.stride(0)), int(l.bt.stride(1)) if l.bt.ndim >= 2 else 0)
                for l in stage.lazy_gdn_entries.values()
                if l.gen == stage._gen
            ),
        )
        cached_plan = stage._prepared_plan
        if cached_plan is not None and cached_plan[0] == bt_key:
            _, undo_tuple, lazy_meta_groups, batched_meta, prep_specs, fallback_recs = cached_plan
            self._launch_tx_kernels(stage, live_bs, v_w, lazy_meta_groups, batched_meta, prep_specs)
            self._sp_sync_batch_commit_after_tx(stage, undo_tuple, live_bs, dev)
            self._batched_undo_meta = (
                {"meta": batched_meta, "live_bs": live_bs, "fallback_recs": fallback_recs}
                if batched_meta is not None
                else None
            )
            self._undo_records = list(undo_tuple)
            self._committed = False
            return

        undo_records: List[_TxUndoRecord] = []
        prep_specs: List[Tuple[Any, ...]] = []
        write_specs: List[Tuple[Any, ...]] = []
        lazy_specs: List[Tuple[Any, ...]] = []
        non_triton_entries: List[Tuple[_TxStageEntry, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, torch.Tensor]] = []

        for entry in stage.entries.values():
            if entry.gen != stage._gen:
                continue
            base = entry.cache.kv_cache_base
            num_blocks = int(base.shape[0])
            tpb = max(1, int(entry.cache.seq_size_per_block))
            bt_live = entry.bt[:live_bs]
            max_blocks = int(bt_live.shape[1]) if bt_live.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0:
                continue

            need_bytes = entry.need_bytes
            base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
            if int(base_u8.shape[1]) < need_bytes:
                raise RuntimeError(
                    f"Opaque pool row too small during commit: {base_u8.shape[1]} < {need_bytes} "
                    f"for tag={entry.tag} layer={entry.layer_idx}"
                )

            phys_slice = entry.phys_ids_buf[: 2 * live_bs]
            commit_slice = entry.commit_u8_buf[: 2 * live_bs]
            undo_slice = entry.undo_u8_buf[: 2 * live_bs]
            rec = _TxUndoRecord(
                cache=entry.cache,
                tag=entry.tag,
                layer_idx=entry.layer_idx,
                phys_ids=phys_slice,
                commit_u8=commit_slice,
                undo_u8=undo_slice,
            )
            undo_records.append(rec)

            if (
                _HAS_TRITON
                and base.is_cuda
                and bt_live.dtype == torch.int32
                and need_bytes % 4 == 0
                and int(base_u8.stride(0)) % 4 == 0
                and int(commit_slice.stride(0)) % 4 == 0
                and int(undo_slice.stride(0)) % 4 == 0
                and base_u8.data_ptr() % 4 == 0
                and commit_slice.data_ptr() % 4 == 0
                and undo_slice.data_ptr() % 4 == 0
            ):
                steps_i32, steps_row_elems = _view_opaque_pool_2d(
                    entry.steps[:live_bs].reshape(live_bs * v_w, -1), torch.int32
                )
                steps_i32_3d = steps_i32.view(live_bs, v_w, steps_row_elems)
                base_i32, _ = _view_opaque_pool_2d(base, torch.int32)
                commit_i32 = commit_slice.view(torch.int32)
                undo_i32 = undo_slice.view(torch.int32)
                n_elems = need_bytes // 4
                spec = (
                    rec,
                    bt_live,
                    steps_i32_3d,
                    base_i32,
                    phys_slice,
                    commit_i32,
                    undo_i32,
                    int(bt_live.stride(0)),
                    int(bt_live.stride(1)),
                    int(steps_i32_3d.stride(0)),
                    int(steps_i32_3d.stride(1)),
                    int(base_i32.stride(0)),
                    int(commit_i32.stride(0)),
                    int(undo_i32.stride(0)),
                    tpb,
                    max_blocks,
                    num_blocks,
                    n_elems,
                )
                prep_specs.append(spec)
                write_specs.append(spec)
            else:
                non_triton_entries.append(
                    (entry, bt_live, phys_slice, commit_slice, undo_slice, need_bytes, tpb, max_blocks, num_blocks, base_u8)
                )

        fallback_recs: List[_TxUndoRecord] = []
        lazy_meta_groups: Optional[List[Dict[str, Any]]] = None
        for lentry in stage.lazy_gdn_entries.values():
            if lentry.gen != stage._gen:
                continue
            base = lentry.cache.kv_cache_base
            num_blocks = int(base.shape[0])
            tpb = max(1, int(lentry.cache.seq_size_per_block))
            bt_live = lentry.bt[:live_bs]
            max_blocks = int(bt_live.shape[1]) if bt_live.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0:
                continue
            phys_slice = lentry.phys_ids_buf[: 2 * live_bs]
            commit_slice = lentry.commit_u8_buf[: 2 * live_bs]
            undo_slice = lentry.undo_u8_buf[: 2 * live_bs]
            rec = _TxUndoRecord(
                cache=lentry.cache,
                tag=lentry.tag,
                layer_idx=lentry.layer_idx,
                phys_ids=phys_slice,
                commit_u8=commit_slice,
                undo_u8=undo_slice,
            )
            undo_records.append(rec)

            base_i32, _ = _view_opaque_pool_2d(base, torch.int32)
            commit_i32 = commit_slice.view(torch.int32)
            undo_i32 = undo_slice.view(torch.int32)
            init_2d = lentry.init_ssm[:live_bs]
            k_4d = lentry.lazy_k[:live_bs]
            v_4d = lentry.lazy_v[:live_bs]
            g_3d = lentry.lazy_g[:live_bs]
            b_3d = lentry.lazy_beta[:live_bs]
            lazy_specs.append(
                (
                    (
                        lentry.num_k_heads,
                        lentry.num_v_heads,
                        lentry.head_k_dim,
                        lentry.head_v_dim,
                        k_4d.dtype,
                    ),
                    bt_live,
                    init_2d,
                    k_4d,
                    v_4d,
                    g_3d,
                    b_3d,
                    base_i32,
                    phys_slice,
                    commit_i32,
                    undo_i32,
                    int(bt_live.stride(0)),
                    int(bt_live.stride(1)),
                    int(init_2d.stride(0)),
                    int(k_4d.stride(0)),
                    int(k_4d.stride(1)),
                    int(k_4d.stride(2)),
                    int(k_4d.stride(3)),
                    int(v_4d.stride(0)),
                    int(v_4d.stride(1)),
                    int(v_4d.stride(2)),
                    int(v_4d.stride(3)),
                    int(g_3d.stride(0)),
                    int(g_3d.stride(1)),
                    int(g_3d.stride(2)),
                    int(b_3d.stride(0)),
                    int(b_3d.stride(1)),
                    int(b_3d.stride(2)),
                    int(base_i32.stride(0)),
                    int(commit_i32.stride(0)),
                    int(undo_i32.stride(0)),
                    tpb,
                    max_blocks,
                    num_blocks,
                )
            )
            full_elems = lentry.ssm_n + lentry.conv_i32_elems
            write_specs.append(
                (
                    rec,
                    bt_live,
                    commit_i32,
                    base_i32,
                    phys_slice,
                    commit_i32,
                    undo_i32,
                    int(bt_live.stride(0)),
                    int(bt_live.stride(1)),
                    0,
                    0,
                    int(base_i32.stride(0)),
                    int(commit_i32.stride(0)),
                    int(undo_i32.stride(0)),
                    tpb,
                    max_blocks,
                    num_blocks,
                    full_elems,
                )
            )
            if lentry.conv_steps is not None and lentry.conv_i32_elems > 0:
                conv_i32, conv_row_elems = _view_opaque_pool_2d(
                    lentry.conv_steps[:live_bs].reshape(live_bs * v_w, -1), torch.int32
                )
                conv_i32_3d = conv_i32.view(live_bs, v_w, conv_row_elems)
                base_conv = base_i32[:, lentry.ssm_n :]
                commit_conv = commit_i32[:, lentry.ssm_n :]
                undo_conv = undo_i32[:, lentry.ssm_n :]
                conv_phys = lentry.conv_phys_buf[: 2 * live_bs]
                prep_specs.append(
                    (
                        rec,
                        bt_live,
                        conv_i32_3d,
                        base_conv,
                        conv_phys,
                        commit_conv,
                        undo_conv,
                        int(bt_live.stride(0)),
                        int(bt_live.stride(1)),
                        int(conv_i32_3d.stride(0)),
                        int(conv_i32_3d.stride(1)),
                        int(base_conv.stride(0)),
                        int(commit_conv.stride(0)),
                        int(undo_conv.stride(0)),
                        tpb,
                        max_blocks,
                        num_blocks,
                        lentry.conv_i32_elems,
                    )
                )

        for (entry, bt_live, phys_slice, commit_slice, undo_slice, need_bytes, tpb, max_blocks, num_blocks, base_u8) in non_triton_entries:
            fallback_recs.append(
                _TxUndoRecord(
                    cache=entry.cache,
                    tag=entry.tag,
                    layer_idx=entry.layer_idx,
                    phys_ids=phys_slice,
                    commit_u8=commit_slice,
                    undo_u8=undo_slice,
                )
            )
            if (
                _HAS_TRITON
                and base_u8.is_cuda
                and need_bytes % 4 == 0
                and int(base_u8.stride(0)) % 4 == 0
                and int(commit_slice.stride(0)) % 4 == 0
                and int(undo_slice.stride(0)) % 4 == 0
                and base_u8.data_ptr() % 4 == 0
                and commit_slice.data_ptr() % 4 == 0
                and undo_slice.data_ptr() % 4 == 0
            ):
                steps_i32, steps_row_elems = _view_opaque_pool_2d(
                    entry.steps[:live_bs].reshape(live_bs * v_w, -1), torch.int32
                )
                steps_i32_3d = steps_i32.view(live_bs, v_w, steps_row_elems)
                base_i32, _ = _view_opaque_pool_2d(entry.cache.kv_cache_base, torch.int32)
                commit_i32 = commit_slice.view(torch.int32)
                undo_i32 = undo_slice.view(torch.int32)
                n_elems = need_bytes // 4
                block_sz = 1024
                _prepare_opaque_tx_commit_triton[(2 * live_bs, int(triton.cdiv(n_elems, block_sz)))](
                    bt_live,
                    stage.seq_before_buf,
                    stage.accept_len_buf,
                    steps_i32_3d,
                    base_i32,
                    phys_slice,
                    commit_i32,
                    undo_i32,
                    int(bt_live.stride(0)),
                    int(bt_live.stride(1)),
                    int(steps_i32_3d.stride(0)),
                    int(steps_i32_3d.stride(1)),
                    int(base_i32.stride(0)),
                    int(commit_i32.stride(0)),
                    int(undo_i32.stride(0)),
                    live_bs,
                    v_w,
                    tpb,
                    max_blocks,
                    num_blocks,
                    n_elems,
                    BLOCK_SIZE=block_sz,
                )
            else:
                a_len = stage.accept_len_buf[:live_bs].to(torch.int64)
                pos_before = stage.seq_before_buf[:live_bs].to(torch.int64)
                valid = a_len > 0
                step_final = (a_len - 1).clamp(min=0, max=max(0, v_w - 1))
                rows_ar = torch.arange(live_bs, dtype=torch.int64, device=dev)
                active_bt = (bt_live != 0).any(dim=1) & valid
                final_pos = (pos_before + a_len - 1).clamp(min=0)
                final_page = (final_pos // tpb).clamp(max=max_blocks - 1)
                final_phys = bt_live.gather(1, final_page.unsqueeze(1)).squeeze(1).to(torch.int32)
                final_phys = torch.where(
                    active_bt & (final_phys > 0) & (final_phys < num_blocks), final_phys, 0
                )
                boundary_step = (tpb - 1) - (pos_before % tpb)
                crossed = (
                    active_bt
                    & ((a_len - 1) > boundary_step)
                    & (boundary_step >= 0)
                    & (boundary_step < v_w)
                )
                bnd_page = (pos_before // tpb).clamp(max=max_blocks - 1)
                bnd_phys = bt_live.gather(1, bnd_page.unsqueeze(1)).squeeze(1).to(torch.int32)
                bnd_phys = torch.where(
                    crossed & (bnd_phys > 0) & (bnd_phys < num_blocks), bnd_phys, 0
                )
                phys_slice[:live_bs].copy_(final_phys)
                phys_slice[live_bs:].copy_(bnd_phys)
                commit_slice[:live_bs].copy_(
                    entry.steps[rows_ar, step_final].contiguous().reshape(live_bs, -1).view(torch.uint8)
                )
                commit_slice[live_bs:].copy_(
                    entry.steps[rows_ar, boundary_step.clamp(min=0, max=max(0, v_w - 1))]
                    .contiguous()
                    .reshape(live_bs, -1)
                    .view(torch.uint8)
                )
                safe_phys = phys_slice.to(torch.int64).clamp(min=0, max=num_blocks - 1)
                undo_slice.copy_(base_u8[:, :need_bytes].index_select(0, safe_phys))

        if lazy_specs:
            lsig = tuple(
                (ls[0], ls[1].data_ptr(), ls[2].data_ptr(), ls[3].data_ptr(), ls[4].data_ptr(), ls[5].data_ptr(), ls[6].data_ptr(), ls[7].data_ptr(), ls[8].data_ptr(), ls[9].data_ptr(), ls[10].data_ptr(), *ls[11:])
                for ls in lazy_specs
            )
            lazy_meta_groups: Optional[List[Dict[str, Any]]] = None
            if stage._lazy_sig == lsig and stage._lazy_meta is not None:
                lazy_meta_groups = stage._lazy_meta
            elif not torch.cuda.is_current_stream_capturing():
                groups_map: Dict[Any, List[Tuple[Any, ...]]] = {}
                for ls in lazy_specs:
                    groups_map.setdefault(ls[0], []).append(ls)
                lazy_meta_groups = []
                for g_key, grp in groups_map.items():
                    lazy_meta_groups.append(
                        (lambda _grp=grp, _gk=g_key: (
                            lambda _f64, _f32: (
                                lambda _d64, _d32, _ne=len(_grp): {
                                    "shape_key": _gk,
                                    "num_e": _ne,
                                    "sample_kv": _grp[0][3],
                                    "ptrs_bt": _d64[0:_ne], "ptrs_init": _d64[_ne:2*_ne],
                                    "ptrs_k": _d64[2*_ne:3*_ne], "ptrs_v": _d64[3*_ne:4*_ne],
                                    "ptrs_g": _d64[4*_ne:5*_ne], "ptrs_beta": _d64[5*_ne:6*_ne],
                                    "ptrs_base": _d64[6*_ne:7*_ne], "ptrs_phys": _d64[7*_ne:8*_ne],
                                    "ptrs_commit": _d64[8*_ne:9*_ne], "ptrs_undo": _d64[9*_ne:10*_ne],
                                    "bt_s0": _d64[10*_ne:11*_ne], "bt_s1": _d64[11*_ne:12*_ne],
                                    "init_sb": _d64[12*_ne:13*_ne], "k_sb": _d64[13*_ne:14*_ne],
                                    "k_sv": _d64[14*_ne:15*_ne], "k_shk": _d64[15*_ne:16*_ne],
                                    "k_sd": _d64[16*_ne:17*_ne], "v_sb": _d64[17*_ne:18*_ne],
                                    "v_sv": _d64[18*_ne:19*_ne], "v_shv": _d64[19*_ne:20*_ne],
                                    "v_vd": _d64[20*_ne:21*_ne], "g_sb": _d64[21*_ne:22*_ne],
                                    "g_sv": _d64[22*_ne:23*_ne], "g_shv": _d64[23*_ne:24*_ne],
                                    "b_sb": _d64[24*_ne:25*_ne], "b_sv": _d64[25*_ne:26*_ne],
                                    "b_shv": _d64[26*_ne:27*_ne], "base_sb": _d64[27*_ne:28*_ne],
                                    "commit_sb": _d64[28*_ne:29*_ne], "undo_sb": _d64[29*_ne:30*_ne],
                                    "tpb": _d32[0:_ne], "max_blocks": _d32[_ne:2*_ne],
                                    "num_blocks": _d32[2*_ne:3*_ne],
                                }
                            )(
                                _pin_stage_copy_to_device(torch.frombuffer(array.array("q", _f64), dtype=torch.int64), dev),
                                _copy_i32_list_to_dev(_f32, dev),
                            )
                        )(
                            [val for k_idx in range(1, 11) for val in [x[k_idx].data_ptr() for x in _grp]]
                            + [val for k_idx in range(11, 31) for val in [x[k_idx] for x in _grp]],
                            [val for k_idx in range(31, 34) for val in [x[k_idx] for x in _grp]],
                        ))()
                    )
                stage._lazy_sig = lsig
                stage._lazy_meta = lazy_meta_groups
            if lazy_meta_groups is None:
                raise RuntimeError("Lazy GDN commit metadata unavailable during CUDA graph capture; run eager warmup first")
        batched_meta: Optional[Dict[str, Any]] = None
        block_sz = 2048
        if len(write_specs) >= 1:
            sig = (
                tuple((s[1].data_ptr(), s[2].data_ptr(), s[3].data_ptr(), s[4].data_ptr(), s[5].data_ptr(), s[6].data_ptr(), *s[7:]) for s in prep_specs),
                tuple((s[3].data_ptr(), s[4].data_ptr(), s[5].data_ptr(), s[6].data_ptr(), s[11], s[12], s[13], s[16], s[17]) for s in write_specs),
            )
            if stage._batched_sig == sig and stage._batched_meta is not None:
                batched_meta = stage._batched_meta
            elif not torch.cuda.is_current_stream_capturing():
                batched_meta = {
                    "prep": _build_batched_tx_meta(prep_specs, block_sz, dev, with_steps=True) if prep_specs else None,
                    "write": _build_batched_tx_meta(write_specs, block_sz, dev, with_steps=False),
                    "block_sz": block_sz,
                }
                stage._batched_sig = sig
                stage._batched_meta = batched_meta
            if batched_meta is None:
                fallback_recs.extend(s[0] for s in write_specs)

        self._launch_tx_kernels(
            stage,
            live_bs,
            v_w,
            lazy_meta_groups if lazy_specs else None,
            batched_meta,
            prep_specs,
        )
        self._sp_sync_batch_commit_after_tx(stage, undo_records, live_bs, dev)
        if not torch.cuda.is_current_stream_capturing() and not non_triton_entries:
            stage._prepared_plan = (
                bt_key,
                tuple(undo_records),
                lazy_meta_groups if lazy_specs else None,
                batched_meta,
                prep_specs,
                list(fallback_recs),
            )

        self._batched_undo_meta = (
            {"meta": batched_meta, "live_bs": live_bs, "fallback_recs": fallback_recs}
            if batched_meta is not None
            else None
        )
        self._undo_records = undo_records
        self._committed = False

    def finish_commit(self, commit: bool) -> None:
        if commit:
            if self._batched_undo_meta is not None:
                wm = self._batched_undo_meta["meta"]["write"]
                lbs = self._batched_undo_meta["live_bs"]
                block_sz = int(self._batched_undo_meta["meta"]["block_sz"])
                if wm["num_tiles"] > 0:
                    self._launch_write_batched(wm, "ptrs_commit", lbs, block_sz)
                for rec in self._batched_undo_meta["fallback_recs"]:
                    _write_opaque_rows_raw_gpu(
                        rec.cache.kv_cache_base, rec.phys_ids, rec.commit_u8, rec.tag, rec.layer_idx
                    )
            else:
                for rec in self._undo_records:
                    _write_opaque_rows_raw_gpu(
                        rec.cache.kv_cache_base, rec.phys_ids, rec.commit_u8, rec.tag, rec.layer_idx
                    )
            self._committed = True
        else:
            if self._committed:
                if self._batched_undo_meta is not None:
                    wm = self._batched_undo_meta["meta"]["write"]
                    lbs = self._batched_undo_meta["live_bs"]
                    block_sz = int(self._batched_undo_meta["meta"]["block_sz"])
                    if wm["num_tiles"] > 0:
                        self._launch_write_batched(wm, "ptrs_undo", lbs, block_sz)
                    for rec in self._batched_undo_meta["fallback_recs"]:
                        _write_opaque_rows_raw_gpu(
                            rec.cache.kv_cache_base, rec.phys_ids, rec.undo_u8, rec.tag, rec.layer_idx
                        )
                else:
                    for rec in self._undo_records:
                        _write_opaque_rows_raw_gpu(
                            rec.cache.kv_cache_base, rec.phys_ids, rec.undo_u8, rec.tag, rec.layer_idx
                        )
            self._committed = False
            self._undo_records.clear()
            self._batched_undo_meta = None
            self._active_pending = None

    def finalize_commit(self) -> None:
        self._undo_records.clear()
        self._batched_undo_meta = None
        self._committed = False
        self._active_pending = None


def _resolve_arch_config(config: Any) -> Qwen4_5ArchConfig:
    arch = getattr(config, "qwen4_5_arch_config", None) or getattr(config, "arch_config", None)
    if arch is not None:
        return arch
    raw = getattr(config, "raw_config", None)
    if isinstance(raw, Mapping):
        return parse_qwen4_5_config_json(raw, strict=False)
    raise ValueError("Cannot resolve Qwen4_5ArchConfig from ModelConfig")


class Qwen4_5RTPModel(_RtpGptModelBase):
    """RTP-LLM ``_py_model_class()`` runtime model supporting Paged KVCache & PD separation."""

    def __init__(
        self,
        config: Any,
        parallelism_config: Any = None,
        weight: Any = None,
        max_generate_batch_size: int = 128,
        fmha_config: Any = None,
        py_hw_kernel_config: Any = None,
        *,
        arch_config: Optional[Qwen4_5ArchConfig] = None,
        backbone: Optional[Qwen4_5Model] = None,
        lm_head: Optional[nn.Linear] = None,
        kv_cache: Any = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if _HAS_COMPILED_GPT_BASE and weight is not None:
            super().__init__(
                config,
                parallelism_config,
                weight,
                max_generate_batch_size,
                fmha_config=fmha_config,
                py_hw_kernel_config=py_hw_kernel_config,
            )
        else:
            nn.Module.__init__(self)

        self.config = config
        if not hasattr(self, "py_hw_kernel_config"):
            self.py_hw_kernel_config = py_hw_kernel_config
        self.parallelism_config = parallelism_config
        self.arch_config = arch_config or _resolve_arch_config(config)
        self.max_generate_batch_size = int(max_generate_batch_size)
        self.kv_cache = kv_cache
        self.lm_head = lm_head
        if not hasattr(self, "_mtp_aux_capture_layer_ids"):
            self._mtp_aux_capture_layer_ids = tuple(
                getattr(config, "capture_aux_hidden_layer_ids", None) or ()
            )
            self._mtp_aux_capture_layer_id_set = frozenset(self._mtp_aux_capture_layer_ids)
            self._mtp_target_hidden_states = None
        self._cpu_target_status = None
        tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1) if parallelism_config is not None else 1
        tp_rank = int(getattr(parallelism_config, "tp_rank", 0) or 0) if parallelism_config is not None else 0
        tp_group = getattr(parallelism_config, "tp_group", None) if parallelism_config is not None else None
        if tp_size > 1 and not bool(getattr(config, "is_mtp", False)):
            self._init_cpu_target_status(parallelism_config)
        if self.arch_config.has_qsa:
            logging.info(
                f"[Qwen4_5 QSA] enabled indexer_head_dim={self.arch_config.indexer_head_dim} "
                f"indexer_n_heads={self.arch_config.indexer_n_heads} "
                f"indexer_kv_heads={self.arch_config.indexer_kv_heads} "
                f"compress_ratio={self.arch_config.indexer_compress_ratio} "
                f"budget={self.arch_config.indexer_budget}"
            )

        ep_size = int(getattr(parallelism_config, "ep_size", 1)) if parallelism_config is not None else 1
        ep_rank = int(getattr(parallelism_config, "ep_rank", 0)) if parallelism_config is not None else 0
        ep_group = getattr(parallelism_config, "ep_group", None) if parallelism_config is not None else None
        raw_sp_size = getattr(parallelism_config, "sp_size", None) if parallelism_config is not None else None
        raw_sp_rank = getattr(parallelism_config, "sp_rank", None) if parallelism_config is not None else None
        sp_group = getattr(parallelism_config, "sp_group", None) if parallelism_config is not None else None
        dp_size = int(getattr(parallelism_config, "dp_size", 1) or 1) if parallelism_config is not None else 1
        dp_rank = int(getattr(parallelism_config, "dp_rank", 0) or 0) if parallelism_config is not None else 0
        dp_group = getattr(parallelism_config, "dp_group", None) if parallelism_config is not None else None
        parallel_mode = getattr(parallelism_config, "parallel_mode", None) if parallelism_config is not None else None
        if parallel_mode is None:
            parallel_mode = getattr(config, "parallel_mode", None)
        pm_norm = str(parallel_mode or "").strip().lower()
        if raw_sp_size is not None and int(raw_sp_size) > 0:
            sp_size = int(raw_sp_size)
        elif pm_norm in ("sp4_tp4_ep4", "tpsp4_ep4", "dp2_tpsp2_ep4") and tp_size > 1:
            sp_size = tp_size
        else:
            sp_size = 1
        if raw_sp_rank is not None:
            sp_rank = int(raw_sp_rank)
        elif sp_size > 1:
            sp_rank = int(tp_rank) % int(sp_size)
        else:
            sp_rank = 0
        raw_dp_attn = getattr(parallelism_config, "dp_attn", None) if parallelism_config is not None else None
        if raw_dp_attn is not None:
            dp_attn = bool(raw_dp_attn)
        elif pm_norm in ("sp4_tp4_ep4", "tpsp4_ep4", "dp2_tpsp2_ep4") or sp_size > 1:
            dp_attn = False
        else:
            dp_attn = dp_size > 1

        if backbone is not None:
            self.model = backbone
        else:
            dev = device or (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            m_params = inspect.signature(Qwen4_5Model.__init__).parameters
            m_kw: Dict[str, Any] = {
                "dtype": dtype,
                "device": dev,
                "ep_size": ep_size,
                "ep_rank": ep_rank,
                "ep_group": ep_group,
            }
            for k, v in (
                ("tp_size", tp_size),
                ("tp_rank", tp_rank),
                ("tp_group", tp_group),
                ("sp_size", sp_size),
                ("sp_rank", sp_rank),
                ("sp_group", sp_group),
                ("dp_size", dp_size),
                ("dp_rank", dp_rank),
                ("dp_group", dp_group),
                ("dp_attn", dp_attn),
                ("parallel_mode", parallel_mode),
            ):
                if k in m_params and (v is not None or k in ("sp_group", "dp_group", "parallel_mode")):
                    if v is not None or k == "parallel_mode":
                        if v is not None:
                            m_kw[k] = v
            self.model = Qwen4_5Model(self.arch_config, **m_kw)
            if weight is not None:
                b_params = inspect.signature(bind_qwen4_5_model_weights).parameters
                b_kw: Dict[str, Any] = {
                    "lm_head": self.lm_head,
                    "ep_size": ep_size,
                    "ep_rank": ep_rank,
                    "ep_group": ep_group,
                }
                if "tp_size" in b_params:
                    b_kw["tp_size"] = tp_size
                    b_kw["tp_rank"] = tp_rank
                    b_kw["tp_group"] = tp_group
                bind_qwen4_5_model_weights(
                    self.model,
                    weight,
                    self.arch_config,
                    **b_kw,
                )

        # Infer effective model TP size from backbone layers (handles TP=1 backbones safely)
        eff_tp = int(getattr(self.model, "tp_size", 1) or 1)
        if eff_tp == 1 and len(self.model.layers) > 0:
            for lyr in self.model.layers:
                if lyr.self_attn is not None and lyr.self_attn.num_heads > 0:
                    eff_tp = max(1, int(self.arch_config.num_attention_heads) // int(lyr.self_attn.num_heads))
                    break
        self.tp_size = eff_tp
        self.tp_rank = (
            int(getattr(self.model, "tp_rank"))
            if (backbone is not None and hasattr(self.model, "tp_rank"))
            else (
                int(tp_rank)
                if parallelism_config is not None
                else int(getattr(self.model, "tp_rank", tp_rank) or 0)
            )
        )
        self.tp_group = tp_group if tp_group is not None else getattr(self.model, "tp_group", None)
        model_sp_size = getattr(self.model, "sp_size", None)
        self.sp_size = int(model_sp_size) if model_sp_size is not None else int(sp_size or 1)
        model_sp_rank = getattr(self.model, "sp_rank", None)
        if model_sp_rank is not None and (backbone is not None or raw_sp_rank is None):
            self.sp_rank = int(model_sp_rank)
        elif raw_sp_rank is not None:
            self.sp_rank = int(raw_sp_rank)
        elif self.sp_size > 1:
            self.sp_rank = int(self.tp_rank) % int(self.sp_size)
        else:
            self.sp_rank = 0
        self.sp_group = sp_group if sp_group is not None else getattr(self.model, "sp_group", None)
        if self.sp_group is None and self.sp_size > 1 and self.sp_size == self.tp_size and self.tp_group is not None:
            self.sp_group = self.tp_group
        self.dp_size = int(getattr(self.model, "dp_size", dp_size) or 1)
        self.dp_rank = (
            int(getattr(self.model, "dp_rank"))
            if (backbone is not None and hasattr(self.model, "dp_rank"))
            else (
                int(dp_rank)
                if parallelism_config is not None
                else int(getattr(self.model, "dp_rank", dp_rank) or 0)
            )
        )
        self.dp_group = dp_group if dp_group is not None else getattr(self.model, "dp_group", None)
        self.dp_attn = bool(getattr(self.model, "dp_attn", dp_attn))
        self.parallel_mode = getattr(self.model, "parallel_mode", None) or parallel_mode
        self.fast_all_reduce = getattr(self.model, "fast_all_reduce", None)
        if (
            self.fast_all_reduce is None
            and self.tp_size > 1
            and self.tp_group is not None
            and getattr(self.lm_head, "weight", None) is not None
            and self.lm_head.weight.device.type == "cuda"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            from rtp_llm.models_py.modules.qwen4_5.fast_comm import get_qwen4_5_fast_all_reduce

            self.fast_all_reduce = get_qwen4_5_fast_all_reduce(
                self.tp_group,
                device=self.lm_head.weight.device,
                dtype=self.lm_head.weight.dtype,
                hidden_dims=(
                    int(self.arch_config.hidden_size),
                    int(self.arch_config.routed_expert_hidden_size),
                ),
            )
        self.bridge = Qwen4_5PagedStateBridge(
            self.arch_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group,
            sp_size=self.sp_size,
            sp_rank=self.sp_rank,
            sp_group=self.sp_group,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            dp_group=self.dp_group,
            dp_attn=self.dp_attn,
            fast_all_reduce=self.fast_all_reduce,
            sp_fast_comm=getattr(self.model, "sp_fast_comm", None),
        )
        self.tx_manager = _Qwen4_5OpaqueTxManager(self.bridge, self.arch_config)
        self._auto_prefill_cg_hits: Dict[Any, int] = {}
        self._auto_prefill_cg_entries: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
        self._auto_prefill_cg_pool: Any = None

    @classmethod
    def from_causal_lm(
        cls,
        causal_lm: Qwen4_5ForCausalLM,
        *,
        model_config: Any = None,
        parallelism_config: Any = None,
        weight: Any = None,
        kv_cache: Any = None,
        max_generate_batch_size: int = 128,
        fmha_config: Any = None,
        py_hw_kernel_config: Any = None,
    ) -> "Qwen4_5RTPModel":
        """Wrap an initialized ``Qwen4_5ForCausalLM`` into an RTP-LLM Paged/PD runtime model."""
        from rtp_llm.config.qwen4_5_config import Qwen4_5ModelConfig

        mcfg = (
            model_config
            if model_config is not None
            else Qwen4_5ModelConfig(arch_config=causal_lm.config)
        )
        return cls(
            config=mcfg,
            parallelism_config=parallelism_config,
            weight=weight,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            arch_config=causal_lm.config,
            backbone=causal_lm.model,
            lm_head=causal_lm.lm_head,
            kv_cache=kv_cache,
            device=causal_lm.model.embed_tokens.weight.device,
            dtype=causal_lm.model.embed_tokens.weight.dtype,
        )

    def initialize(self, init_resource: Any) -> bool:
        """Attach engine-allocated ``KVCache`` pool from ``PyModelInitResources``."""
        self.kv_cache = getattr(init_resource, "kv_cache", None)
        for attr in ("py_hw_kernel_config", "hw_kernel_config", "server_config"):
            if getattr(self, attr, None) is None:
                val = getattr(init_resource, attr, None)
                if val is not None:
                    setattr(self, attr, val)
        return True

    def prepare_fmha_impl(
        self,
        inputs: Any,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Any:
        del cuda_graph_selection_mode
        if self.kv_cache is None:
            return None
        flat_ids = getattr(inputs, "input_ids", None)
        if flat_ids is None or flat_ids.device.type != "cuda":
            return None
        attn_by_tag = getattr(inputs, "attention_inputs_by_tag", None)
        attn_inputs = (
            attn_by_tag
            if isinstance(attn_by_tag, Mapping) and len(attn_by_tag) > 0
            else getattr(inputs, "attention_inputs", None)
        )
        if attn_inputs is None or (isinstance(attn_inputs, Mapping) and len(attn_inputs) == 0):
            return None
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        is_target_verify = bool(getattr(primary, "is_target_verify", False))
        is_prefill = not is_target_verify and bool(getattr(primary, "is_prefill", False))
        if is_prefill:
            if not is_cuda_graph:
                return None
            raw_in = getattr(primary, "input_lengths", None)
            if not isinstance(raw_in, torch.Tensor) or raw_in.numel() == 0:
                return None
            in_lens = tuple(int(x) for x in raw_in.tolist())
            if sum(in_lens) <= 0:
                return None
            raw_pfx = getattr(primary, "prefix_lengths", None)
            pfx_lens = (
                tuple(int(x) for x in raw_pfx.tolist())
                if isinstance(raw_pfx, torch.Tensor) and raw_pfx.numel() == len(in_lens)
                else (0,) * len(in_lens)
            )
            plan = self.bridge._build_ragged_prefill_plan(in_lens, pfx_lens, flat_ids.device, freeze=True)
            distinct_tpbs = {
                max(1, int(c.seq_size_per_block))
                for lid in range(len(self.model.layers))
                for c in self.kv_cache.get_layer_cache_groups(lid)
            }
            for tpb_c in distinct_tpbs:
                _ = self.bridge._get_snap_schedule(plan, tpb_c, flat_ids.device)
            pos_ext = getattr(inputs, "combo_position_ids", None)
            if pos_ext is None or (isinstance(pos_ext, torch.Tensor) and pos_ext.numel() == 0):
                pos_ext = getattr(primary, "combo_position_ids", None)
            flat_pos = (
                pos_ext.to(device=flat_ids.device, dtype=torch.int64)
                if isinstance(pos_ext, torch.Tensor) and pos_ext.numel() > 0
                else plan.tok_abs_pos_i64
            )
            pre_handle = Qwen4_5PrefillHandle(
                self.arch_config,
                primary,
                in_lens=in_lens,
                pfx_lens=pfx_lens,
                plan=plan,
                flat_pos=flat_pos,
                is_cuda_graph=True,
            )
            eff_sp_p, eff_rk_p, _, _ = self.bridge._resolve_sp_dist()
            if eff_sp_p > 1 and plan.total_tokens > 0:
                sub_plan_p = self.bridge._build_ragged_sp_subplan(plan, eff_sp_p, eff_rk_p, flat_ids.device)
                for tpb_c in distinct_tpbs:
                    _ = self.bridge._get_sp_snap_base(plan, sub_plan_p, tpb_c, flat_ids.device)
                    _ = self.bridge._get_sp_sync_slots(plan, tpb_c, True, eff_sp_p, flat_ids.device)
                    _ = self.bridge._get_sp_sync_slots(plan, tpb_c, False, eff_sp_p, flat_ids.device)
                has_cs_writer = (
                    isinstance(attn_inputs, Mapping)
                    and any(getattr(g, "cache_store_writer", None) is not None for g in attn_inputs.values())
                ) or (
                    not isinstance(attn_inputs, Mapping)
                    and getattr(attn_inputs, "cache_store_writer", None) is not None
                )
                self.bridge._prewarm_prefill_sp_sync_meta(
                    self.model,
                    self.kv_cache,
                    attn_inputs,
                    plan,
                    sub_plan_p,
                    pre_handle,
                    sync_all_blocks=has_cs_writer,
                    sp_size=eff_sp_p,
                    sp_rank=eff_rk_p,
                    device=flat_ids.device,
                )
            return pre_handle
        if not is_target_verify and not is_prefill and is_cuda_graph:
            eff_sp_d, _, _, _ = self.bridge._resolve_sp_dist()
            bsz_d = int(flat_ids.numel())
            if eff_sp_d > 1 and bsz_d >= eff_sp_d and (bsz_d % eff_sp_d) == 0:
                self.bridge._prewarm_dec_sp_meta(self.model, self.kv_cache, attn_inputs, bsz_d, flat_ids.device)
        if not is_target_verify and (
            isinstance(self.kv_cache, LocalKVCache) and not self.arch_config.has_qsa
        ):
            return None

        full_caches = [
            c
            for lid in range(len(self.model.layers))
            for c in self.kv_cache.get_layer_cache_groups(lid)
            if str(c.tag) == FULL_ATTN_KV_TAG
        ]
        if not full_caches:
            return None
        tpb = int(full_caches[0].seq_size_per_block)
        if is_target_verify:
            idx_kv_in = _select_group_for_tag(attn_inputs, INDEXER_KV_TAG) if self.arch_config.has_qsa else None
            idx_st_in = _select_group_for_tag(attn_inputs, INDEXER_STATE_TAG) if self.arch_config.has_qsa else None
            return Qwen4_5TargetVerifyHandle(
                self.arch_config,
                primary,
                idx_kv_in,
                idx_st_in,
                page_size=tpb,
                is_cuda_graph=is_cuda_graph,
            )

        if self.arch_config.has_qsa:
            idx_kv_in = _select_group_for_tag(attn_inputs, INDEXER_KV_TAG)
            idx_st_in = _select_group_for_tag(attn_inputs, INDEXER_STATE_TAG)
            return Qwen4_5QSADecodeHandle(
                self.arch_config,
                primary,
                idx_kv_in,
                idx_st_in,
                page_size=tpb,
                is_cuda_graph=is_cuda_graph,
            )

        from rtp_llm.ops import AttentionConfigs, KvCacheDataType

        attn_cfg = AttentionConfigs()
        attn_cfg.head_num = int(self.arch_config.num_attention_heads) // int(getattr(self, "tp_size", 1))
        attn_cfg.kv_head_num = int(self.arch_config.num_key_value_heads)
        attn_cfg.size_per_head = int(self.arch_config.head_dim)
        attn_cfg.tokens_per_block = tpb
        attn_cfg.kernel_tokens_per_block = tpb
        attn_cfg.dtype = self.model.embed_tokens.weight.dtype
        src_attn = getattr(self.config, "attn_config", None)
        if src_attn is not None and getattr(src_attn, "kv_cache_dtype", None) is not None:
            attn_cfg.kv_cache_dtype = src_attn.kv_cache_dtype
        else:
            attn_cfg.kv_cache_dtype = KvCacheDataType.BASE
        attn_cfg.is_causal = True
        attn_cfg.use_mla = False
        attn_cfg.need_rope_kv_cache = False
        return Qwen4_5DecodeAttnHandle(attn_cfg, primary, is_cuda_graph=is_cuda_graph)

    def _init_cpu_target_status(self, parallelism_config: Any) -> None:
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
        from rtp_llm.models_py.distributed.cpu_status import create_cpu_tp_status
        from rtp_llm.models_py.modules.qwen4_exp.cpu_target_status import _GlooTargetStatus

        group = _get_group(Group.TP)
        world_size = dist.get_world_size(group)
        if _HAS_COMPILED_GPT_BASE:
            from rtp_llm.ops.compute_ops import rtp_llm_ops
            import datetime, os, socket
            from pathlib import Path

            status_size = getattr(rtp_llm_ops, "cpu_status_size", None)
            native_available = callable(status_size) and bool(status_size(world_size))
            identity = (
                socket.gethostname(),
                Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                os.getuid(),
                os.stat("/dev/shm").st_dev,
                native_available,
            )
            peers = [None] * world_size
            dist.all_gather_object(peers, identity, group=group)
            if native_available and all(peer == identity for peer in peers):
                self._cpu_target_status = create_cpu_tp_status(group, rtp_llm_ops)
                return
        ranks = [dist.get_global_rank(group, r) for r in range(world_size)]
        import datetime
        control = dist.new_group(
            ranks=ranks,
            backend="gloo",
            timeout=datetime.timedelta(seconds=60),
            use_local_synchronization=True,
        )
        self._cpu_target_status = _GlooTargetStatus(control)

    def has_mtp_hidden_buffer(self) -> bool:
        return bool(self._mtp_aux_capture_layer_ids)

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        hidden = getattr(self, "_mtp_target_hidden_states", None)
        if hidden is None:
            return None
        if num_tokens < 0 or num_tokens == int(hidden.shape[0]):
            return hidden
        if num_tokens > int(hidden.shape[0]):
            raise ValueError(
                f"Requested {num_tokens} MTP target hidden rows, got {int(hidden.shape[0])}"
            )
        return hidden[:num_tokens]

    def requires_speculative_target_commit(self) -> bool:
        return True

    def has_speculative_target_status_reducer(self) -> bool:
        return self._cpu_target_status is not None

    def reduce_speculative_target_prepare_status(self, local_ok: bool) -> bool:
        if self._cpu_target_status is None:
            return bool(local_ok)
        return bool(self._cpu_target_status.reduce(local_ok))

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
        self.tx_manager.prepare_commit(accept_len)

    def finish_speculative_target_commit(self, commit: bool) -> None:
        self.tx_manager.finish_commit(bool(commit))

    def finalize_speculative_target_commit(self) -> None:
        self.tx_manager.finalize_commit()

    def _extract_verify_seq_before(
        self, attn_inputs: Any, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        raw_pfx = getattr(primary, "prefix_lengths_device", None)
        pfx = raw_pfx() if callable(raw_pfx) else raw_pfx
        if not isinstance(pfx, torch.Tensor) or pfx.numel() == 0:
            pfx = getattr(primary, "prefix_lengths", None)
        if not isinstance(pfx, torch.Tensor) or pfx.numel() == 0:
            pfx = getattr(primary, "sequence_lengths", None)
        if not isinstance(pfx, torch.Tensor) or pfx.numel() == 0:
            return torch.zeros((batch_size,), dtype=torch.int32, device=device)
        if pfx.device == device and pfx.dtype == torch.int32:
            return pfx[:batch_size]
        return _pin_stage_copy_to_device(pfx[:batch_size].contiguous().to(dtype=torch.int32), device)

    def _resolve_replay_attn(self, inputs: Any) -> Any:
        attn_by_tag = getattr(inputs, "attention_inputs_by_tag", None)
        if isinstance(attn_by_tag, Mapping) and len(attn_by_tag) > 0:
            return attn_by_tag
        return getattr(inputs, "attention_inputs", None)

    def on_cuda_graph_warmup(self, inputs: Any) -> None:
        del inputs
        self.tx_manager.on_cuda_graph_warmup()

    def on_cuda_graph_capture(self, graph_key: int, inputs: Any) -> None:
        del graph_key, inputs
        self.tx_manager.on_cuda_graph_capture()

    def on_cuda_graph_replay(self, graph_key: int, inputs: Any) -> None:
        attn_inputs = self._resolve_replay_attn(inputs)
        if attn_inputs is None:
            self.tx_manager.on_cuda_graph_warmup()
            return
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        if not bool(getattr(primary, "is_target_verify", False)):
            self.tx_manager.on_cuda_graph_warmup()
            return
        flat_ids = inputs.input_ids.reshape(-1)
        bsz = int(graph_key)
        verify_w = max(1, int(flat_ids.numel()) // bsz)
        seq_before_dev = self._extract_verify_seq_before(attn_inputs, bsz, flat_ids.device)
        self.tx_manager.on_cuda_graph_replay(bsz, verify_w, seq_before_dev, attn_inputs)

    def _wrap_outputs(
        self, hidden: torch.Tensor, aux_hidden: Optional[torch.Tensor] = None
    ) -> Any:
        self._mtp_target_hidden_states = aux_hidden
        if _RtpPyModelOutputs is not None and _HAS_COMPILED_GPT_BASE:
            if aux_hidden is not None:
                return _RtpPyModelOutputs(hidden, aux_hidden)
            return _RtpPyModelOutputs(hidden)
        return LocalPyModelOutputs(
            hidden_states=hidden, mtp_target_hidden_states=aux_hidden
        )

    def _build_local_positions(
        self,
        attn_grp: Any,
        input_lengths: List[int],
        prefix_lengths: List[int],
        seq_lens_before: List[int],
        is_prefill: bool,
        device: torch.device,
        inputs: Any,
    ) -> torch.Tensor:
        pos = getattr(inputs, "combo_position_ids", None)
        if pos is None or (isinstance(pos, torch.Tensor) and pos.numel() == 0):
            pos = getattr(attn_grp, "combo_position_ids", None)
        if isinstance(pos, torch.Tensor) and pos.numel() > 0:
            return pos.to(device=device, dtype=torch.int64)
        if is_prefill:
            chunks = [
                torch.arange(p, p + ln, device=device, dtype=torch.int64)
                for p, ln in zip(prefix_lengths, input_lengths)
            ]
            return torch.cat(chunks, dim=0)
        return torch.tensor(seq_lens_before, device=device, dtype=torch.int64).unsqueeze(1)

    def _collect_auto_cg_kv_state(
        self, attn_inputs: Any, bsz: int
    ) -> Optional[Tuple[Tuple[Tuple[int, int], ...], Dict[str, Tuple[torch.Tensor, int]], Set[int]]]:
        pool_ptrs: List[Tuple[int, int]] = []
        tag_info: Dict[str, Tuple[torch.Tensor, int]] = {}
        distinct_tpbs: Set[int] = set()
        for lid in range(len(self.model.layers)):
            for c in self.kv_cache.get_layer_cache_groups(lid):
                sc_base = getattr(c, "kv_scale_base", None)
                pool_ptrs.append((int(c.kv_cache_base.data_ptr()), int(sc_base.data_ptr()) if sc_base is not None else 0))
                tag = str(c.tag)
                tpb_c = max(1, int(c.seq_size_per_block))
                distinct_tpbs.add(tpb_c)
                if tag not in tag_info:
                    bt = _block_table_for_tag(attn_inputs, tag)
                    if not bt.is_cuda or bt.dtype != torch.int32 or bt.ndim != 2 or int(bt.shape[0]) < bsz:
                        return None
                    tag_info[tag] = (bt, tpb_c)
        return tuple(pool_ptrs), tag_info, distinct_tpbs

    @staticmethod
    def _update_auto_prefill_cg_buffers(
        entry: Dict[str, Any],
        flat_ids: torch.Tensor,
        flat_pos: Optional[torch.Tensor],
        pfx_lens: Sequence[int],
        tag_info: Mapping[str, Tuple[torch.Tensor, int]],
    ) -> None:
        bsz = int(entry["bsz"])
        in_len = int(entry["in_len"])
        entry["static_ids"].copy_(flat_ids)
        pfx_t = _copy_i32_list_to_dev(pfx_lens, flat_ids.device)
        entry["static_prefix_i32"].copy_(pfx_t)
        entry["static_start_pos_i64"].copy_(entry["static_prefix_i32"])
        torch.add(
            entry["static_prefix_i32"].view(bsz, 1),
            entry["base_offsets_0_i32"],
            out=entry["static_tok_abs_pos_i32"].view(bsz, in_len),
        )
        entry["static_tok_abs_pos_i64"].copy_(entry["static_tok_abs_pos_i32"])
        if flat_pos is not None:
            entry["static_pos"].copy_(flat_pos)
        else:
            torch.add(
                entry["static_start_pos_i64"].view(bsz, 1),
                entry["base_offsets_0"],
                out=entry["static_pos"].view(bsz, in_len),
            )
        torch.add(
            entry["static_prefix_i32"].view(bsz, 1),
            entry["base_offsets_1"],
            out=entry["static_vis_q"],
        )
        for tag, (bt, _) in tag_info.items():
            s_bt = entry["static_bts"][tag]
            cap_cols = int(s_bt.shape[1])
            in_cols = int(bt.shape[1])
            n_copy = min(cap_cols, in_cols)
            if n_copy > 0:
                s_bt[:, :n_copy].copy_(bt[:bsz, :n_copy])
            if n_copy < cap_cols:
                s_bt[:, n_copy:].zero_()

    def release_auto_prefill_cg(self) -> None:
        """Reset and drop captured Auto-Prefill CUDA graphs cleanly before PG teardown."""
        entries = getattr(self, "_auto_prefill_cg_entries", None)
        if entries:
            for ent in list(entries.values()):
                g = ent.get("graph") if isinstance(ent, dict) else None
                if g is not None:
                    try:
                        g.reset()
                    except Exception:
                        pass
            entries.clear()
        if hasattr(self, "_auto_prefill_cg_pool"):
            self._auto_prefill_cg_pool = None

    def __del__(self) -> None:
        if sys is not None and not getattr(sys, "is_finalizing", lambda: True)():
            try:
                self.release_auto_prefill_cg()
            except Exception:
                pass

    @staticmethod
    def _parse_cuda_graph_flag(val: Any) -> Optional[bool]:
        """Parse boolean-like env/config flags, returning ``None`` if unset or blank."""
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        norm = str(val).strip().lower()
        if not norm:
            return None
        if norm in ("0", "false", "no", "off"):
            return False
        if norm in ("1", "true", "yes", "on"):
            return True
        return None

    def _is_auto_prefill_cg_enabled(self) -> bool:
        """Gate Auto-Prefill-CG by explicit env override and engine/model CUDA Graph config."""
        explicit = self._parse_cuda_graph_flag(os.environ.get("QWEN4_5_AUTO_PREFILL_CG"))
        if explicit is not None:
            return explicit
        if self._parse_cuda_graph_flag(os.environ.get("ENABLE_CUDA_GRAPH")) is False:
            return False
        cfg_obj = getattr(self, "config", None)
        for src in (
            getattr(self, "py_hw_kernel_config", None),
            getattr(self, "hw_kernel_config", None),
            getattr(self, "server_config", None),
            cfg_obj,
            getattr(cfg_obj, "py_hw_kernel_config", None),
            getattr(cfg_obj, "hw_kernel_config", None),
            getattr(cfg_obj, "server_config", None),
        ):
            if src is None:
                continue
            raw_val = (
                src.get("enable_cuda_graph", None)
                if isinstance(src, Mapping)
                else getattr(src, "enable_cuda_graph", None)
            )
            if self._parse_cuda_graph_flag(raw_val) is False:
                return False
        return True

    def _maybe_run_auto_prefill_cg(
        self,
        flat_ids: torch.Tensor,
        flat_pos: Optional[torch.Tensor],
        in_lens: Sequence[int],
        pfx_lens: Sequence[int],
        attn_inputs: Any,
        writers: Mapping[str, Any],
        enable_yoco_decode: Optional[bool],
        cap_ids: Sequence[int],
        device: torch.device,
    ) -> Optional[Any]:
        if not self._is_auto_prefill_cg_enabled():
            return None
        mxfp4_backend = os.environ.get("QWEN4_5_MXFP4_BACKEND", "auto").strip().lower()
        if mxfp4_backend == "ref":
            return None
        strict_bf16 = os.environ.get("QWEN4_5_MXFP4_STRICT_BF16", "0").strip()
        if (
            device.type != "cuda"
            or not _HAS_TRITON
            or self.kv_cache is None
            or torch.cuda.is_current_stream_capturing()
            or self.dp_size != 1
            or not self.arch_config.has_qsa
        ):
            return None
        bsz = len(in_lens)
        if bsz <= 0 or bsz > 4 or len(pfx_lens) != bsz:
            return None
        if len(set(in_lens)) != 1 or len(set(pfx_lens)) != 1 or int(in_lens[0]) <= 0:
            return None
        if not all(int(p) > 0 for p in pfx_lens):
            return None
        min_tok = int(os.environ.get("QWEN4_5_AUTO_PREFILL_CG_MIN_TOKENS", "32"))
        max_tok = int(os.environ.get("QWEN4_5_AUTO_PREFILL_CG_MAX_TOKENS", "512"))
        total_tok = int(flat_ids.numel())
        if total_tok < min_tok or total_tok > max_tok:
            return None
        eff_sp, eff_rk, _, _ = self.bridge._resolve_sp_dist()
        if eff_sp > 1 and total_tok < eff_sp:
            return None
        in_len = int(in_lens[0])
        t_pfx = int(pfx_lens[0])
        t_tot = t_pfx + in_len
        if t_tot > 131072:
            return None
        if any(bool(getattr(g, "is_s_padded", False)) for _, g in _iter_attention_groups(attn_inputs)):
            return None

        kv_state = self._collect_auto_cg_kv_state(attn_inputs, bsz)
        if kv_state is None:
            return None
        pool_ptrs, tag_info, distinct_tpbs = kv_state
        lcm_tpb = 64
        for tpb_c in distinct_tpbs:
            lcm_tpb = math.lcm(lcm_tpb, int(tpb_c))
        ctx_bucket = 128
        while ctx_bucket < t_tot:
            ctx_bucket *= 2
        if ctx_bucket > 131072:
            return None

        dev_idx = int(device.index if device.index is not None else -1)
        cg_key = (
            dev_idx,
            tuple(int(x) for x in in_lens),
            tuple(int(p) % lcm_tpb for p in pfx_lens),
            ctx_bucket,
            bool(writers) if eff_sp > 1 else False,
            enable_yoco_decode,
            tuple(int(x) for x in cap_ids),
            mxfp4_backend,
            strict_bf16,
        )

        entry = self._auto_prefill_cg_entries.get(cg_key)
        if entry is not None:
            can_replay = entry["pool_ptrs"] == pool_ptrs and all(
                int(bt.shape[1]) <= int(entry["static_bts"][tag].shape[1])
                for tag, (bt, _) in tag_info.items()
            )
            if can_replay:
                self._update_auto_prefill_cg_buffers(entry, flat_ids, flat_pos, pfx_lens, tag_info)
                entry["graph"].replay()
                if writers:
                    for lid_p in range(len(self.model.layers)):
                        publish_qwen4_5_layer_caches(self.kv_cache, lid_p, writers)
                self._auto_prefill_cg_entries.move_to_end(cg_key)
                s_aux = entry["static_aux"]
                return self._wrap_outputs(
                    entry["static_out"].clone(),
                    s_aux.clone() if s_aux is not None else None,
                )
            self._auto_prefill_cg_entries.pop(cg_key, None)

        min_hits = int(os.environ.get("QWEN4_5_AUTO_PREFILL_CG_MIN_HITS", "2"))
        prev_hits = self._auto_prefill_cg_hits.get(cg_key, 0)
        if prev_hits < 0:
            return None
        hits = prev_hits + 1
        self._auto_prefill_cg_hits[cg_key] = hits
        if hits < min_hits:
            return None

        # Capture guarded CUDA graph with bucketed static block tables & dynamic prefix offset
        try:
            from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import _select_top_tokens

            static_ids = torch.empty_like(flat_ids)
            static_pos = torch.empty((total_tok,), dtype=torch.int64, device=device)
            static_vis_q = torch.empty((bsz, in_len), dtype=torch.int32, device=device)
            base_offsets_0 = torch.arange(in_len, dtype=torch.int64, device=device).view(1, in_len)
            base_offsets_1 = base_offsets_0.to(torch.int32) + 1
            plan = self.bridge._build_ragged_prefill_plan(in_lens, pfx_lens, device, freeze=True)
            in_len_cpu = torch.tensor(in_lens, dtype=torch.int32)
            pfx_len_cpu = torch.tensor(pfx_lens, dtype=torch.int32)
            seq_len_cpu = in_len_cpu + pfx_len_cpu
            static_bts: Dict[str, torch.Tensor] = {}
            wrapped_ai: Dict[str, LocalPyAttentionInputs] = {}
            for tag, (bt, tpb_c) in tag_info.items():
                cap_cols = max(int(bt.shape[1]), (ctx_bucket + tpb_c - 1) // tpb_c + 1)
                s_bt = torch.zeros((bsz, cap_cols), dtype=torch.int32, device=device)
                static_bts[tag] = s_bt
                wrapped_ai[tag] = LocalPyAttentionInputs(
                    is_prefill=True,
                    input_lengths=in_len_cpu,
                    sequence_lengths=seq_len_cpu,
                    prefix_lengths=pfx_len_cpu,
                    kv_cache_block_id=s_bt,
                    kv_cache_block_id_device=s_bt,
                    kv_cache_kernel_block_id=s_bt,
                    kv_cache_kernel_block_id_device=s_bt,
                    is_cuda_graph=True,
                    total_tokens=total_tok,
                )
            for tpb_c in distinct_tpbs:
                _ = self.bridge._get_snap_schedule(plan, tpb_c, device)
                _ = self.bridge._get_snap_rel_pages(plan, tpb_c, device)
            if eff_sp > 1 and plan.total_tokens > 0:
                sub_plan_p = self.bridge._build_ragged_sp_subplan(plan, eff_sp, eff_rk, device)
                for tpb_c in distinct_tpbs:
                    _ = self.bridge._get_sp_snap_base(plan, sub_plan_p, tpb_c, device)
                    _ = self.bridge._get_sp_sync_slots(plan, tpb_c, True, eff_sp, device)
                    _ = self.bridge._get_sp_sync_slots(plan, tpb_c, False, eff_sp, device)

            # Pre-compile exact QSA TopK specialization for bucketed cols before capture
            seen_qsa: Set[Tuple[int, int]] = set()
            qsa_rows = (bsz * (in_len if eff_sp <= 1 else (in_len + eff_sp - 1) // eff_sp))
            for lyr in self.model.layers:
                sa_m = getattr(lyr, "self_attn", None)
                idx_m = getattr(sa_m, "indexer", None) if sa_m is not None else None
                if idx_m is not None:
                    qk = (int(idx_m.compress_ratio), int(idx_m.token_budget))
                    if qk not in seen_qsa:
                        seen_qsa.add(qk)
                        cols_p = max(1, ctx_bucket // qk[0])
                        _ = _select_top_tokens(
                            torch.zeros((qsa_rows, cols_p), dtype=torch.float32, device=device),
                            torch.zeros((qsa_rows,), dtype=torch.int32, device=device),
                            compress_ratio=qk[0],
                            token_budget=qk[1],
                            scores_are_masked=True,
                        )

            static_handle = Qwen4_5PrefillHandle(
                self.arch_config,
                wrapped_ai[FULL_ATTN_KV_TAG],
                in_lens=tuple(int(x) for x in in_lens),
                pfx_lens=tuple(int(x) for x in pfx_lens),
                plan=plan,
                flat_pos=static_pos,
                is_cuda_graph=True,
                is_auto_cg_dynamic_pfx=True,
                static_vis_q=static_vis_q,
                cg_max_ctx_tokens=ctx_bucket,
                sp_sync_all_blocks_override=bool(writers),
            )
            new_entry: Dict[str, Any] = {
                "bsz": bsz,
                "in_len": in_len,
                "pool_ptrs": pool_ptrs,
                "static_ids": static_ids,
                "static_pos": static_pos,
                "static_prefix_i32": plan.prefix_i32,
                "static_start_pos_i64": plan.start_pos_i64,
                "static_tok_abs_pos_i32": plan.tok_abs_pos_i32,
                "static_tok_abs_pos_i64": plan.tok_abs_pos_i64,
                "static_vis_q": static_vis_q,
                "base_offsets_0": base_offsets_0,
                "base_offsets_0_i32": base_offsets_0.to(torch.int32),
                "base_offsets_1": base_offsets_1,
                "static_bts": static_bts,
            }
            self._update_auto_prefill_cg_buffers(new_entry, flat_ids, flat_pos, pfx_lens, tag_info)
            if eff_sp > 1 and plan.total_tokens > 0:
                self.bridge._prewarm_prefill_sp_sync_meta(
                    self.model,
                    self.kv_cache,
                    wrapped_ai,
                    plan,
                    sub_plan_p,
                    static_handle,
                    sync_all_blocks=bool(writers),
                    sp_size=eff_sp,
                    sp_rank=eff_rk,
                    device=device,
                )

            if self._auto_prefill_cg_pool is None:
                self._auto_prefill_cg_pool = torch.cuda.graph_pool_handle()
            eff_ep_size = int(getattr(self.model, "ep_size", 1) or 1)
            eff_ep_group = getattr(self.model, "ep_group", None)
            need_consensus = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and (self.tp_size > 1 or self.sp_size > 1 or eff_ep_size > 1)
            )
            cons_grp = (
                eff_ep_group
                if eff_ep_size > 1
                else (self.tp_group if self.tp_group is not None else self.sp_group)
            )
            graph = torch.cuda.CUDAGraph()
            cap_exc: Optional[BaseException] = None
            res_cap: Any = None
            try:
                # Thread-local capture prevents poisoning background threads (e.g. CrcBlockCopy.cu)
                # that poll cudaStreamQuery / cudaEventQuery on separate streams during capture.
                with torch.cuda.graph(
                    graph,
                    pool=self._auto_prefill_cg_pool,
                    capture_error_mode="thread_local",
                ):
                    res_cap = self.bridge.run_prefill_ragged(
                        self.model,
                        static_ids,
                        static_pos,
                        in_lens,
                        pfx_lens,
                        self.kv_cache,
                        wrapped_ai,
                        {},
                        prefill_handle=static_handle,
                        enable_yoco_decode=enable_yoco_decode,
                        capture_layer_ids=cap_ids,
                    )
            except (RuntimeError, torch.AcceleratorError) as exc:
                cap_exc = exc
                try:
                    graph.reset()
                except Exception:
                    pass
                del graph
                if not need_consensus:
                    raise
            if need_consensus:
                ok_t = torch.tensor([0 if cap_exc is not None else 1], dtype=torch.int32, device=device)
                torch.distributed.all_reduce(ok_t, op=torch.distributed.ReduceOp.MIN, group=cons_grp)
                if int(ok_t.item()) == 0:
                    if cap_exc is None:
                        try:
                            graph.reset()
                        except Exception:
                            pass
                        del graph
                    self._auto_prefill_cg_hits[cg_key] = -1
                    return None
            if cap_exc is not None:
                raise cap_exc
            graph.replay()
            if isinstance(res_cap, tuple):
                static_out, static_aux = res_cap[0], res_cap[1]
            else:
                static_out, static_aux = res_cap, None
            new_entry["graph"] = graph
            new_entry["static_out"] = static_out
            new_entry["static_aux"] = static_aux
            new_entry["static_handle"] = static_handle
            new_entry["wrapped_ai"] = wrapped_ai

            if writers:
                for lid_p in range(len(self.model.layers)):
                    publish_qwen4_5_layer_caches(self.kv_cache, lid_p, writers)

            max_graphs = int(os.environ.get("QWEN4_5_AUTO_PREFILL_CG_MAX_GRAPHS", "16"))
            self._auto_prefill_cg_entries[cg_key] = new_entry
            while len(self._auto_prefill_cg_entries) > max(1, max_graphs):
                self._auto_prefill_cg_entries.popitem(last=False)
            return self._wrap_outputs(
                static_out.clone(),
                static_aux.clone() if static_aux is not None else None,
            )
        except (RuntimeError, torch.AcceleratorError):
            self._auto_prefill_cg_hits[cg_key] = -1
            return None

    @torch.no_grad()
    def forward(
        self,
        inputs: Any,
        fmha_impl: Any = None,
        *,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Any:
        """Execute RTP-LLM Paged/PD forward returning ``PyModelOutputs(hidden_states)``."""
        attn_by_tag = getattr(inputs, "attention_inputs_by_tag", None)
        if isinstance(attn_by_tag, Mapping) and len(attn_by_tag) > 0:
            attn_inputs: Any = attn_by_tag
        else:
            attn_inputs = getattr(inputs, "attention_inputs", None)
        flat_ids = inputs.input_ids.reshape(-1)
        device = flat_ids.device
        cap_ids = self._mtp_aux_capture_layer_ids
        is_empty_attn = attn_inputs is None or (
            isinstance(attn_inputs, Mapping) and len(attn_inputs) == 0
        )

        def _empty_dp_outputs() -> Any:
            dtype_e = self.model.embed_tokens.weight.dtype
            eff_sp_e, _, _, _ = self.bridge._resolve_sp_dist()
            self.bridge._run_empty_ep_collectives(
                self.model, dtype_e, device, sp_active=(eff_sp_e > 1)
            )
            empty_h = torch.empty(
                (0, self.arch_config.hidden_size), dtype=dtype_e, device=device
            )
            empty_aux = (
                torch.empty(
                    (0, len(cap_ids) * self.arch_config.hidden_size),
                    dtype=dtype_e,
                    device=device,
                )
                if cap_ids
                else None
            )
            return self._wrap_outputs(empty_h, empty_aux)

        if is_empty_attn:
            if int(flat_ids.numel()) == 0:
                if self.kv_cache is None and self.dp_size <= 1:
                    hidden_e = torch.zeros(
                        (0, self.arch_config.hidden_size),
                        dtype=self.model.embed_tokens.weight.dtype,
                        device=device,
                    )
                    aux_e = (
                        torch.zeros(
                            (0, len(cap_ids) * self.arch_config.hidden_size),
                            dtype=hidden_e.dtype,
                            device=device,
                        )
                        if cap_ids
                        else None
                    )
                    return self._wrap_outputs(hidden_e, aux_e)
                with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()):
                    return _empty_dp_outputs()
            raise RuntimeError("PyModelInputs.attention_inputs is required")

        # Pick primary attention group (prefer "full" if present)
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        is_target_verify = bool(getattr(primary, "is_target_verify", False))
        is_prefill = bool(getattr(primary, "is_prefill", False)) and not is_target_verify

        if self.kv_cache is None and not is_prefill:
            hidden = torch.zeros(
                (int(flat_ids.numel()), self.arch_config.hidden_size),
                dtype=self.model.embed_tokens.weight.dtype,
                device=device,
            )
            aux = (
                torch.zeros(
                    (int(flat_ids.numel()), len(cap_ids) * self.arch_config.hidden_size),
                    dtype=hidden.dtype,
                    device=device,
                )
                if cap_ids
                else None
            )
            return self._wrap_outputs(hidden, aux)

        has_cs_writer = (
            isinstance(attn_inputs, Mapping)
            and any(getattr(g, "cache_store_writer", None) is not None for g in attn_inputs.values())
        ) or (not isinstance(attn_inputs, Mapping) and getattr(attn_inputs, "cache_store_writer", None) is not None)
        writers = (
            prepare_qwen4_5_cache_store_writers(
                attn_inputs, self.kv_cache, layer_count=len(self.model.layers)
            )
            if is_prefill and self.kv_cache is not None and has_cs_writer and int(flat_ids.numel()) > 0
            else {}
        )

        dev_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with dev_ctx:
            if is_target_verify:
                if not _HAS_TRITON or device.type != "cuda" or self.kv_cache is None:
                    raise RuntimeError(
                        "Qwen4_5 target verification requires CUDA + Triton + KVCache"
                    )
                if int(flat_ids.numel()) == 0:
                    return _empty_dp_outputs()
                raw_pfx = getattr(primary, "prefix_lengths", None)
                if isinstance(raw_pfx, torch.Tensor) and raw_pfx.device == device and raw_pfx.numel() > 0:
                    pfx_dev = raw_pfx
                else:
                    raw_pfx_fn = getattr(primary, "prefix_lengths_device", None)
                    pfx_dev = raw_pfx_fn() if callable(raw_pfx_fn) else raw_pfx_fn
                    if not isinstance(pfx_dev, torch.Tensor) or pfx_dev.numel() == 0:
                        pfx_dev = raw_pfx
                if not isinstance(pfx_dev, torch.Tensor) or pfx_dev.numel() == 0:
                    raise RuntimeError("Target verification requires prefix_lengths")
                bsz = int(pfx_dev.numel())
                total_toks = int(flat_ids.numel())
                if bsz <= 0 or total_toks % bsz != 0:
                    raise ValueError(
                        f"Target verify total tokens {total_toks} not divisible by batch_size {bsz}"
                    )
                verify_w = total_toks // bsz
                seq_before_dev = (
                    pfx_dev.to(dtype=torch.int32)
                    if pfx_dev.device == device
                    else _pin_stage_copy_to_device(pfx_dev.contiguous().to(dtype=torch.int32).reshape(-1), device)
                )
                ids_2d = flat_ids.view(bsz, verify_w)
                pos = getattr(inputs, "combo_position_ids", None)
                if pos is None or (isinstance(pos, torch.Tensor) and pos.numel() == 0):
                    pos = getattr(primary, "combo_position_ids", None)
                if isinstance(pos, torch.Tensor) and pos.numel() == total_toks:
                    pos_2d = pos.to(device=device, dtype=torch.int64).view(bsz, verify_w)
                else:
                    pos_2d = (
                        seq_before_dev.to(torch.int64).unsqueeze(1)
                        + torch.arange(verify_w, device=device, dtype=torch.int64).unsqueeze(0)
                    )
                handle = (
                    fmha_impl
                    if isinstance(fmha_impl, Qwen4_5TargetVerifyHandle)
                    else self.prepare_fmha_impl(inputs, is_cuda_graph=False)
                )
                if handle is None:
                    raise RuntimeError("Target verification failed to prepare Qwen4_5TargetVerifyHandle")
                tx_stage = self.tx_manager.acquire_stage(
                    bsz, verify_w, is_cuda_graph=bool(handle.is_cuda_graph)
                )
                res = self.bridge._run_target_verify_batch_gpu(
                    self.model,
                    ids_2d,
                    pos_2d,
                    self.kv_cache,
                    attn_inputs,
                    seq_lens_before_dev=seq_before_dev,
                    verify_handle=handle,
                    tx_stage=tx_stage,
                    enable_yoco_decode=enable_yoco_decode,
                    capture_layer_ids=cap_ids,
                )
                if isinstance(res, tuple):
                    return self._wrap_outputs(res[0], res[1])
                return self._wrap_outputs(res)

            if is_prefill:
                if int(flat_ids.numel()) == 0:
                    return _empty_dp_outputs()
                pre_handle = fmha_impl if isinstance(fmha_impl, Qwen4_5PrefillHandle) else None
                if pre_handle is not None:
                    in_lens = pre_handle.in_lens
                    pfx_lens = pre_handle.pfx_lens
                    flat_pos = pre_handle.flat_pos
                else:
                    in_lens = primary.input_lengths.tolist()
                    pfx_lens = (
                        primary.prefix_lengths.tolist()
                        if getattr(primary, "prefix_lengths", None) is not None
                        and primary.prefix_lengths.numel() == len(in_lens)
                        else [0] * len(in_lens)
                    )
                    pos_ext = getattr(inputs, "combo_position_ids", None)
                    if pos_ext is None or (isinstance(pos_ext, torch.Tensor) and pos_ext.numel() == 0):
                        pos_ext = getattr(primary, "combo_position_ids", None)
                    flat_pos = (
                        pos_ext.to(device=device, dtype=torch.int64)
                        if isinstance(pos_ext, torch.Tensor) and pos_ext.numel() > 0
                        else None
                    )
                if pre_handle is None:
                    cg_out = self._maybe_run_auto_prefill_cg(
                        flat_ids,
                        flat_pos,
                        in_lens,
                        pfx_lens,
                        attn_inputs,
                        writers,
                        enable_yoco_decode,
                        cap_ids,
                        device,
                    )
                    if cg_out is not None:
                        return cg_out
                res = self.bridge.run_prefill_ragged(
                    self.model,
                    flat_ids,
                    flat_pos,
                    in_lens,
                    pfx_lens,
                    self.kv_cache,
                    attn_inputs,
                    writers,
                    prefill_handle=pre_handle,
                    enable_yoco_decode=enable_yoco_decode,
                    capture_layer_ids=cap_ids,
                )
                if isinstance(res, tuple):
                    return self._wrap_outputs(res[0], res[1])
                return self._wrap_outputs(res)

            if int(flat_ids.numel()) == 0:
                return _empty_dp_outputs()

            if (
                device.type == "cuda"
                and self.kv_cache is not None
                and (
                    not isinstance(self.kv_cache, LocalKVCache)
                    or isinstance(fmha_impl, Qwen4_5QSADecodeHandle)
                )
                and _HAS_TRITON
            ):
                bsz = int(flat_ids.numel())
                ids_2d = flat_ids.view(bsz, 1)
                seq_plus_1_dev = getattr(primary, "sequence_lengths_plus_1_device", None)
                if (
                    seq_plus_1_dev is not None
                    and seq_plus_1_dev.is_cuda
                    and int(seq_plus_1_dev.numel()) == bsz
                ):
                    seq_before_dev = seq_plus_1_dev.to(torch.int32) - 1
                else:
                    seq_src = primary.sequence_lengths
                    seq_before_dev = (
                        seq_src.to(dtype=torch.int32)
                        if seq_src.device == device
                        else _pin_stage_copy_to_device(seq_src.contiguous().to(dtype=torch.int32).reshape(-1), device)
                    )
                pos = getattr(inputs, "combo_position_ids", None)
                if pos is None or (isinstance(pos, torch.Tensor) and pos.numel() == 0):
                    pos = getattr(primary, "combo_position_ids", None)
                if isinstance(pos, torch.Tensor) and pos.numel() == bsz:
                    pos_2d = pos.to(device=device, dtype=torch.int64).view(bsz, 1)
                else:
                    pos_2d = seq_before_dev.to(torch.int64).view(bsz, 1)
                handle = (
                    fmha_impl
                    if isinstance(
                        fmha_impl, (Qwen4_5DecodeAttnHandle, Qwen4_5QSADecodeHandle)
                    )
                    else self.prepare_fmha_impl(inputs, is_cuda_graph=False)
                )
                res = self.bridge.run_decode_batch(
                    self.model,
                    ids_2d,
                    pos_2d,
                    self.kv_cache,
                    attn_inputs,
                    writers,
                    seq_lens_before_dev=seq_before_dev,
                    decode_attn_handle=handle,
                    enable_yoco_decode=enable_yoco_decode,
                    capture_layer_ids=cap_ids,
                )
                if isinstance(res, tuple):
                    return self._wrap_outputs(res[0], res[1])
                return self._wrap_outputs(res)

            seq_before = [int(x) for x in primary.sequence_lengths.tolist()]
            bsz = len(seq_before)
            ids_2d = flat_ids.view(bsz, 1)
            pos_2d = self._build_local_positions(
                primary, [1] * bsz, [0] * bsz, seq_before, False, device, inputs
            ).view(bsz, 1)
            res = self.bridge.run_decode_batch(
                self.model,
                ids_2d,
                pos_2d,
                self.kv_cache,
                attn_inputs,
                writers,
                seq_lens_before=seq_before,
                enable_yoco_decode=enable_yoco_decode,
                capture_layer_ids=cap_ids,
            )
            if isinstance(res, tuple):
                return self._wrap_outputs(res[0], res[1])
            return self._wrap_outputs(res)

    @torch.no_grad()
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("lm_head is not attached to Qwen4_5RTPModel")
        dev = hidden_states.device
        with (torch.cuda.device(dev) if dev.type == "cuda" else nullcontext()):
            logits = self.lm_head(hidden_states)
            if (
                self.tp_size > 1
                and hasattr(self.lm_head, "weight")
                and int(self.lm_head.weight.shape[0]) < int(self.arch_config.vocab_size)
            ):
                grp = self.tp_group
                if (
                    grp is None
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                    and int(torch.distributed.get_world_size()) == int(self.tp_size)
                ):
                    grp = torch.distributed.group.WORLD
                if grp is not None:
                    ws = int(torch.distributed.get_world_size(grp))
                    if ws > 1:
                        fast_ar = getattr(self, "fast_all_reduce", None)
                        if fast_ar is not None and fast_ar.group is grp:
                            logits = fast_ar.all_gather(logits, dim=-1)
                        else:
                            chunks = [torch.empty_like(logits) for _ in range(ws)]
                            torch.distributed.all_gather(chunks, logits.contiguous(), group=grp)
                            logits = torch.cat(chunks, dim=-1)
                        vocab_size = int(self.arch_config.vocab_size)
                        if int(logits.shape[-1]) > vocab_size:
                            logits = logits[..., :vocab_size]
            return logits


# ---------------------------------------------------------------------------
# End-to-End PD Disaggregated Helper (Prefill -> CacheStore -> Decode -> Text)
# ---------------------------------------------------------------------------


class SimpleByteTokenizer:
    """Zero-dependency UTF-8 byte tokenizer fallback when no HF tokenizer is mounted."""

    def __init__(self, vocab_size: int = 2048, *, bos_id: int = 1, eos_id: int = 2) -> None:
        self.vocab_size = int(vocab_size)
        self.bos_token_id = int(bos_id)
        self.eos_token_id = int(eos_id)
        self._byte_offset = 16

    def encode(self, text: str) -> List[int]:
        raw = text.encode("utf-8", errors="replace")
        return [self.bos_token_id] + [self._byte_offset + b for b in raw]

    def decode(self, ids: Sequence[int]) -> str:
        bs: List[int] = []
        for tid in ids:
            t = int(tid)
            if t == self.eos_token_id or t == self.bos_token_id:
                continue
            if self._byte_offset <= t < self._byte_offset + 256:
                bs.append(t - self._byte_offset)
            else:
                # Map arbitrary model-generated token IDs into readable printable ASCII/UTF-8 marker
                bs.append(ord("a") + (t % 26))
        return bytes(bs).decode("utf-8", errors="replace")


def build_block_tables_for_requests(
    kv_cache: LocalKVCache,
    lengths: Sequence[int],
    *,
    existing_tables: Optional[Dict[str, List[List[int]]]] = None,
) -> Dict[str, List[List[int]]]:
    """Allocate sufficient physical blocks per tag so ``max(lengths)`` tokens fit cleanly."""
    tpb = kv_cache.tokens_per_block
    result: Dict[str, List[List[int]]] = {}
    for tag in kv_cache.group_tags():
        tables = (
            [list(row) for row in existing_tables[tag]]
            if existing_tables is not None and tag in existing_tables
            else [[] for _ in lengths]
        )
        for b, total_len in enumerate(lengths):
            need_blocks = max(1, (total_len + tpb - 1) // tpb)
            if len(tables[b]) < need_blocks:
                tables[b].extend(kv_cache.allocate_blocks(tag, need_blocks - len(tables[b])))
        result[tag] = tables
    return result


def _pack_tables_flat_into(tables: Sequence[Sequence[int]], flat: List[int]) -> Tuple[int, int]:
    rows = len(tables)
    max_cols = max((len(r) for r in tables), default=1)
    if max_cols == 0:
        max_cols = 1
    if rows == 0:
        return 0, max_cols
    for row in tables:
        ln_r = len(row)
        flat.extend(row)
        if ln_r < max_cols:
            flat.extend([0] * (max_cols - ln_r))
    return rows, max_cols


def pack_padded_block_tensor(tables: List[List[int]], device: torch.device) -> torch.Tensor:
    max_cols = max((len(r) for r in tables), default=1)
    if max_cols == 0:
        max_cols = 1
    rows = len(tables)
    if rows == 0:
        return torch.zeros((0, max_cols), dtype=torch.int32, device=device)
    flat: List[int] = []
    _pack_tables_flat_into(tables, flat)
    arr = array.array("i", flat)
    cpu_flat = torch.frombuffer(arr, dtype=torch.int32)
    if device.type == "cpu":
        return cpu_flat.clone().view(rows, max_cols)
    return _pin_stage_copy_to_device(cpu_flat, device).view(rows, max_cols)


def make_tagged_attention_inputs(
    tables_by_tag: Dict[str, List[List[int]]],
    *,
    is_prefill: bool,
    input_lengths: Sequence[int],
    sequence_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    device: torch.device,
    cache_store_writer: Any = None,
    req_block_ids_for_store: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, LocalPyAttentionInputs]:
    in_list = [int(x) for x in input_lengths]
    seq_list = [int(x) for x in sequence_lengths]
    pfx_list = [int(x) for x in prefix_lengths]
    total_tok = sum(in_list)

    flat_all: List[int] = []

    def _append_seg(vals: Sequence[int]) -> Tuple[int, int]:
        a = len(flat_all)
        flat_all.extend(vals)
        b = len(flat_all)
        pad = (-len(vals)) & 3
        if pad:
            flat_all.extend([0] * pad)
        return a, b

    in_s, in_e = _append_seg(in_list)
    seq_s, seq_e = _append_seg(seq_list)
    pfx_s, pfx_e = _append_seg(pfx_list)

    tag_spans: Dict[str, Tuple[int, int, int, int]] = {}
    dedup_tables: Dict[Tuple[Tuple[int, ...], ...], Tuple[int, int, int, int]] = {}
    for tag, tbl in tables_by_tag.items():
        sig = tuple(tuple(int(x) for x in row) for row in tbl)
        hit = dedup_tables.get(sig)
        if hit is None:
            a = len(flat_all)
            rows, cols = _pack_tables_flat_into(sig, flat_all)
            b = len(flat_all)
            pad = (-(b - a)) & 3
            if pad:
                flat_all.extend([0] * pad)
            hit = (a, b, rows, cols)
            dedup_tables[sig] = hit
        tag_spans[tag] = hit

    if flat_all:
        arr_cpu = array.array("i", flat_all)
        cpu_flat = torch.frombuffer(arr_cpu, dtype=torch.int32)
        dev_flat = (
            _pin_stage_copy_to_device(cpu_flat, device)
            if device.type != "cpu"
            else cpu_flat
        )
    else:
        cpu_flat = torch.empty((0,), dtype=torch.int32, device="cpu")
        dev_flat = torch.empty((0,), dtype=torch.int32, device=device)

    in_len_t = cpu_flat[in_s:in_e]
    seq_len_t = cpu_flat[seq_s:seq_e]
    pfx_len_t = cpu_flat[pfx_s:pfx_e]
    in_len_d = dev_flat[in_s:in_e] if in_e > in_s else torch.empty((0,), dtype=torch.int32, device=device)
    seq_len_d = dev_flat[seq_s:seq_e] if seq_e > seq_s else torch.empty((0,), dtype=torch.int32, device=device)
    pfx_len_d = dev_flat[pfx_s:pfx_e] if pfx_e > pfx_s else torch.empty((0,), dtype=torch.int32, device=device)
    in_ver = int(in_len_t._version)
    pfx_ver = int(pfx_len_t._version)

    groups: Dict[str, LocalPyAttentionInputs] = {}
    for tag, tbl in tables_by_tag.items():
        a, b, rows, cols = tag_spans[tag]
        if rows == 0:
            tbl_cpu = torch.zeros((0, cols), dtype=torch.int32, device="cpu")
            tbl_dev = torch.zeros((0, cols), dtype=torch.int32, device=device)
        else:
            tbl_cpu = cpu_flat[a:b].view(rows, cols)
            tbl_dev = dev_flat[a:b].view(rows, cols)
        cs_in = (
            LocalPyCacheStoreInputs(
                tag=tag,
                block_ids=tuple(req_block_ids_for_store.get(tag, ())),
            )
            if cache_store_writer is not None and req_block_ids_for_store is not None
            else (
                LocalPyCacheStoreInputs(
                    tag=tag,
                    block_ids=tuple(sorted({bid for row in tbl for bid in row if bid > 0})),
                )
                if cache_store_writer is not None
                else None
            )
        )
        groups[tag] = LocalPyAttentionInputs(
            is_prefill=is_prefill,
            input_lengths=in_len_t,
            sequence_lengths=seq_len_t,
            prefix_lengths=pfx_len_t,
            kv_cache_block_id=tbl_cpu,
            kv_cache_block_id_device=tbl_dev,
            kv_cache_kernel_block_id=tbl_cpu,
            kv_cache_kernel_block_id_device=tbl_dev,
            cache_store_inputs=cs_in,
            cache_store_writer=cache_store_writer,
            total_tokens=total_tok,
            _input_lengths_dev=in_len_d,
            _input_lengths_src_ver=in_ver,
            _prefix_lengths_dev=pfx_len_d,
            _prefix_lengths_src_ver=pfx_ver,
            _sequence_lengths_dev=seq_len_d,
        )
    return groups


def transfer_published_blocks_to_decode(
    published: Mapping[Tuple[int, str], Tuple[Tuple[int, ...], Optional[torch.Tensor]]],
    src_tables_by_tag: Dict[str, List[List[int]]],
    decode_kv: LocalKVCache,
    lengths: Sequence[int],
) -> Dict[str, List[List[int]]]:
    """Transfer all CacheStore-published physical blocks from Prefill P to Decode D."""
    dst_tables = build_block_tables_for_requests(decode_kv, lengths)
    # Build src_phys_block -> dst_phys_block mapping per tag
    id_map_by_tag: Dict[str, Dict[int, int]] = {}
    for tag, src_rows in src_tables_by_tag.items():
        m: Dict[int, int] = {}
        dst_rows = dst_tables[tag]
        for s_row, d_row in zip(src_rows, dst_rows):
            for sb, db in zip(s_row, d_row):
                m[int(sb)] = int(db)
        id_map_by_tag[tag] = m

    dst_dev = decode_kv.device
    if dst_dev.type == "cuda":
        torch.cuda.synchronize(dst_dev)
    synced_src: set[torch.device] = set()
    with (torch.cuda.device(dst_dev) if dst_dev.type == "cuda" else nullcontext()):
        for (lid, tag), (src_blocks, payload) in published.items():
            assert payload is not None
            if payload.device.type == "cuda" and payload.device not in synced_src:
                torch.cuda.synchronize(payload.device)
                synced_src.add(payload.device)
            dst_layer_cache = decode_kv.get_layer_cache(lid, tag)
            id_map = id_map_by_tag[tag]
            payload_on_dst = payload.to(dst_layer_cache.kv_cache_base.device, non_blocking=False)
            for i, sb in enumerate(src_blocks):
                if sb in id_map:
                    dst_layer_cache.kv_cache_base[id_map[sb]].copy_(payload_on_dst[i])

    return dst_tables


class PDDisaggregatedSession:
    """Orchestrates end-to-end Prefill(P) -> CacheStore Transfer -> Decode(D) token generation."""

    def __init__(
        self,
        prefill_model: Qwen4_5RTPModel,
        decode_model: Qwen4_5RTPModel,
        *,
        tokens_per_block: int = 16,
        max_blocks_per_tag: int = 256,
        enable_yoco_decode: Optional[bool] = None,
    ) -> None:
        self.p_model = prefill_model
        self.d_model = decode_model
        self.tpb = int(tokens_per_block)
        self.enable_yoco_decode = enable_yoco_decode
        self.plan = build_qwen4_5_kv_cache_plan(
            prefill_model.arch_config, enable_yoco_decode=enable_yoco_decode
        )
        p_dev = prefill_model.model.embed_tokens.weight.device
        d_dev = decode_model.model.embed_tokens.weight.device
        dt = prefill_model.model.embed_tokens.weight.dtype

        num_blocks = {t: max_blocks_per_tag for t in QWEN4_5_CACHE_TAGS}
        self.p_kv = LocalKVCache(
            self.plan, tokens_per_block=self.tpb, num_blocks_per_tag=num_blocks, device=p_dev, dtype=dt
        )
        self.d_kv = LocalKVCache(
            self.plan, tokens_per_block=self.tpb, num_blocks_per_tag=num_blocks, device=d_dev, dtype=dt
        )
        self.p_model.kv_cache = self.p_kv
        self.d_model.kv_cache = self.d_kv
        self.writer = LocalMemoryCacheStoreWriter(record_payload=True)

    @torch.no_grad()
    def generate(
        self,
        prompt_token_lists: Sequence[Sequence[int]],
        *,
        max_new_tokens: int = 16,
        prefix_lengths: Optional[Sequence[int]] = None,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """Run ragged PD Prefill on P, transfer published KV+SSM+ShortConv+PLE to D, decode on D."""
        bsz = len(prompt_token_lists)
        in_lens = [len(p) for p in prompt_token_lists]
        pfx_lens = list(prefix_lengths) if prefix_lengths is not None else [0] * bsz
        tot_lens = [p + i for p, i in zip(pfx_lens, in_lens)]

        p_dev = self.p_model.model.embed_tokens.weight.device
        d_dev = self.d_model.model.embed_tokens.weight.device

        # Reserve physical pages up to total_len + max_new_tokens on both P (only tot_lens needed) and D
        p_tables = build_block_tables_for_requests(self.p_kv, tot_lens)
        flat_ids = torch.tensor(
            [tok for seq in prompt_token_lists for tok in seq],
            dtype=torch.int64,
            device=p_dev,
        )
        self.writer.clear()
        p_attn = make_tagged_attention_inputs(
            p_tables,
            is_prefill=True,
            input_lengths=in_lens,
            sequence_lengths=tot_lens,
            prefix_lengths=pfx_lens,
            device=p_dev,
            cache_store_writer=self.writer,
        )
        yoco_flag = (
            self.enable_yoco_decode if enable_yoco_decode is None else enable_yoco_decode
        )
        p_in = LocalPyModelInputs(input_ids=flat_ids, attention_inputs=p_attn)
        p_out = self.p_model.forward(p_in, enable_yoco_decode=yoco_flag)

        # Extract last-token hidden state per ragged request -> first token logits on P
        last_indices: List[int] = []
        acc = 0
        for ln in in_lens:
            acc += ln
            last_indices.append(acc - 1)
        last_h = p_out.hidden_states[torch.tensor(last_indices, dtype=torch.long, device=p_dev)]
        first_logits = self.p_model.compute_logits(last_h)  # [B, V]
        first_tokens = torch.argmax(first_logits, dim=-1).tolist()

        # Transfer all CacheStore-published layers/tags from P -> D and pre-allocate decode growth room
        d_max_lens = [t + max_new_tokens for t in tot_lens]
        d_tables = transfer_published_blocks_to_decode(
            self.writer.published, p_tables, self.d_kv, tot_lens
        )
        d_tables = build_block_tables_for_requests(
            self.d_kv, d_max_lens, existing_tables=d_tables
        )

        gen_tokens: List[List[int]] = [[int(first_tokens[b])] for b in range(bsz)]
        step_logits: List[torch.Tensor] = [first_logits.detach().cpu()]
        if max_new_tokens <= 1:
            return gen_tokens, step_logits

        # Decode remaining steps on independent Decode instance D
        cur_lens = list(tot_lens)
        next_toks = list(first_tokens)
        for _ in range(max_new_tokens - 1):
            dec_ids = torch.tensor(next_toks, dtype=torch.int64, device=d_dev)
            d_attn = make_tagged_attention_inputs(
                d_tables,
                is_prefill=False,
                input_lengths=[1] * bsz,
                sequence_lengths=cur_lens,
                prefix_lengths=[0] * bsz,
                device=d_dev,
                cache_store_writer=None,
            )
            d_in = LocalPyModelInputs(input_ids=dec_ids, attention_inputs=d_attn)
            d_out = self.d_model.forward(d_in, enable_yoco_decode=yoco_flag)
            logits = self.d_model.compute_logits(d_out.hidden_states)  # [B, V]
            step_logits.append(logits.detach().cpu())
            next_toks = torch.argmax(logits, dim=-1).tolist()
            for b in range(bsz):
                gen_tokens[b].append(int(next_toks[b]))
                cur_lens[b] += 1

        return gen_tokens, step_logits


__all__ = [
    "LocalKVCache",
    "LocalLayerKVCache",
    "LocalMemoryCacheStoreWriter",
    "LocalPyAttentionInputs",
    "LocalPyCacheStoreInputs",
    "LocalPyModelInputs",
    "LocalPyModelOutputs",
    "PDDisaggregatedSession",
    "QWEN4_5_CACHE_TAGS",
    "QWEN4_5_SIDE_TAGS",
    "Qwen4_5PagedStateBridge",
    "Qwen4_5PrefillHandle",
    "Qwen4_5QSADecodeHandle",
    "Qwen4_5RTPModel",
    "SimpleByteTokenizer",
    "build_block_tables_for_requests",
    "make_tagged_attention_inputs",
    "pack_padded_block_tensor",
    "prepare_qwen4_5_cache_store_writers",
    "publish_qwen4_5_layer_caches",
    "transfer_published_blocks_to_decode",
    "validate_qwen4_5_pd_cache_inputs",
]
