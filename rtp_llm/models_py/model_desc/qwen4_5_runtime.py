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

import copy
import logging
import math
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
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import bind_qwen4_5_model_weights
from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
    build_qsa_neox_cos_sin,
    qsa_select_causal_from_projected,
    score_and_select_qsa_paged,
    view_indexer_kv_pool,
    view_indexer_state_pool,
    write_qsa_decode_cache_gpu_,
    write_qsa_prefill_cache,
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
    ) -> None:
        self.plan = plan
        self.tokens_per_block = int(tokens_per_block)
        self.device = device
        self.dtype = dtype
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
                    dtype=dtype,
                    device=device,
                )
                self._add_cache(lid, FULL_ATTN_KV_TAG, base)
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

    def _add_cache(self, layer_idx: int, tag: str, base: torch.Tensor) -> None:
        self.layer_caches[layer_idx][tag] = LocalLayerKVCache(
            kv_cache_base=base,
            seq_size_per_block=self.tokens_per_block,
            layer_id=layer_idx,
            tag=tag,
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

    def input_lengths_device(self) -> torch.Tensor:
        return self.input_lengths.to(self.kv_cache_block_id_device.device)

    def prefix_lengths_device(self) -> torch.Tensor:
        if self.prefix_lengths.device == self.kv_cache_block_id_device.device:
            return self.prefix_lengths
        return self.prefix_lengths.to(self.kv_cache_block_id_device.device)


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
    elem_size = torch.empty((), dtype=target_dtype).element_size()
    need_bytes = math.prod(target_shape) * elem_size
    row_u8 = base[phys_id].reshape(-1).view(torch.uint8)
    if int(row_u8.numel()) < need_bytes:
        raise RuntimeError(
            f"Opaque pool row too small: {row_u8.numel()} bytes < {need_bytes} bytes "
            f"for shape={target_shape} dtype={target_dtype}"
        )
    return row_u8[:need_bytes].view(target_dtype).view(*target_shape)


def _load_single_req_full_attn_kv(
    base: torch.Tensor,
    blocks: torch.Tensor,
    total_tokens: int,
    tokens_per_block: int,
    *,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    allow_zero_padded: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read [total_tokens, H_kv, D] K/V from MHA paged pool (NHD, HND, or 2D packed)."""
    if num_kv_heads is not None and head_dim is not None:
        nhd = _view_mha_kv_nhd(base, num_kv_heads, tokens_per_block, head_dim)
    else:
        nhd = base
        num_kv_heads, head_dim = int(nhd.shape[3]), int(nhd.shape[4])
    if total_tokens <= 0:
        empty = torch.empty((0, num_kv_heads, head_dim), dtype=nhd.dtype, device=nhd.device)
        return empty, empty
    n_pages = (total_tokens + tokens_per_block - 1) // tokens_per_block
    page_ids = blocks[:n_pages].to(device=nhd.device, dtype=torch.long)
    if allow_zero_padded and (page_ids == 0).all().item():
        z = torch.zeros((total_tokens, num_kv_heads, head_dim), dtype=nhd.dtype, device=nhd.device)
        return z, z.clone()
    if (page_ids <= 0).any().item() or (page_ids >= nhd.shape[0]).any().item():
        raise RuntimeError(
            f"Invalid FullAttn physical page ids {page_ids.tolist()} for pool={nhd.shape[0]}"
        )
    gathered = nhd.index_select(0, page_ids)  # [n_pages, 2, B_tok, H_kv, D]
    k = gathered[:, 0].reshape(n_pages * tokens_per_block, num_kv_heads, head_dim)[:total_tokens]
    v = gathered[:, 1].reshape(n_pages * tokens_per_block, num_kv_heads, head_dim)[:total_tokens]
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
) -> None:
    """Write [T_new, H_kv, D] K/V starting at ``start_pos`` into paged pool."""
    if num_kv_heads is not None and head_dim is not None:
        nhd = _view_mha_kv_nhd(base, num_kv_heads, tokens_per_block, head_dim)
    else:
        nhd = base
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
        phys_id = _check_phys_block(raw_phys, int(nhd.shape[0]), FULL_ATTN_KV_TAG, -1)
        take = min(tokens_per_block - in_page, t_new - written)
        nhd[phys_id, 0, in_page : in_page + take].copy_(k[written : written + take])
        nhd[phys_id, 1, in_page : in_page + take].copy_(v[written : written + take])
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
    def _store_decode_full_attn_kv_triton(
        pool_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        seq_before_ptr,
        s_blk: tl.int64,
        s_kv: tl.int64,
        s_h: tl.int64,
        s_tok: tl.int64,
        s_d: tl.int64,
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
        tl.store(pool_ptr + dst_base, k_val, mask=mask_d)
        tl.store(pool_ptr + dst_base + s_kv, v_val, mask=mask_d)


def _store_decode_full_attn_kv_gpu(
    kv_cache_layer: Any,
    block_table: torch.Tensor,
    seq_before_dev: torch.Tensor,
    k_4d: torch.Tensor,
    v_4d: torch.Tensor,
) -> None:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU paged KV store")
    from rtp_llm.models_py.modules.factory.attention.common import (
        reshape_paged_kv_cache,
    )

    tpb = int(kv_cache_layer.seq_size_per_block)
    bsz, _, h_kv, head_dim = k_4d.shape
    max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if bsz == 0 or max_blocks == 0:
        return
    pool_hnd = reshape_paged_kv_cache(kv_cache_layer.kv_cache_base, h_kv, tpb, head_dim)
    if int(pool_hnd.shape[2]) != h_kv or int(pool_hnd.shape[3]) != tpb:
        pool_hnd = pool_hnd.transpose(2, 3)
    num_blocks = int(pool_hnd.shape[0])
    k_3d = k_4d[:, 0]
    v_3d = v_4d[:, 0]
    s_blk, s_kv, s_h, s_tok, s_d = pool_hnd.stride()
    block_d = int(triton.next_power_of_2(head_dim))
    _store_decode_full_attn_kv_triton[(bsz, h_kv)](
        pool_hnd,
        k_3d,
        v_3d,
        block_table,
        seq_before_dev,
        s_blk,
        s_kv,
        s_h,
        s_tok,
        s_d,
        int(k_3d.stride(0)),
        int(k_3d.stride(1)),
        int(k_3d.stride(2)),
        int(v_3d.stride(0)),
        int(v_3d.stride(1)),
        int(v_3d.stride(2)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        tpb,
        max_blocks,
        num_blocks,
        head_dim,
        BLOCK_D=block_d,
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
    from rtp_llm.models_py.modules.factory.attention.common import (
        reshape_paged_kv_cache,
    )

    bsz, head_num, _ = q_3d.shape
    tpb = int(kv_cache_layer.seq_size_per_block)
    max_blocks = int(bt_kv.shape[1]) if bt_kv.ndim >= 2 else 0
    if bsz == 0 or max_blocks == 0:
        return torch.zeros_like(q_3d)
    pool_hnd = reshape_paged_kv_cache(kv_cache_layer.kv_cache_base, kv_heads, tpb, head_dim)
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
    _write_opaque_rows_triton[grid](
        dst_view,
        src_view,
        phys_i32,
        int(dst_view.stride(0)),
        int(src_view.stride(0)),
        n_elems,
        num_blocks,
        BLOCK_SIZE=block_sz,
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
            output_gate=None,
            is_cuda_graph=self.is_cuda_graph,
        )
        return out.squeeze(2)


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
            output_gate=None,
            is_cuda_graph=self.is_cuda_graph,
        )
        return out.squeeze(2)


class Qwen4_5PagedStateBridge:
    """Adapts RTP-LLM paged ``KVCache`` pools <-> ``Qwen4_5DecoderLayer`` execution."""

    def __init__(self, arch_cfg: Qwen4_5ArchConfig) -> None:
        self.cfg = arch_cfg
        self.ssm_shape = (
            arch_cfg.linear_num_value_heads,
            arch_cfg.linear_key_head_dim,
            arch_cfg.linear_value_head_dim,
        )
        k_dim = arch_cfg.linear_num_key_heads * arch_cfg.linear_key_head_dim
        v_dim = arch_cfg.linear_num_value_heads * arch_cfg.linear_value_head_dim
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

        pos = seq_before_dev.to(torch.int64)
        read_page = ((pos - 1).clamp(min=0) // tpb).clamp(max=max_blocks - 1)
        read_phys = block_table.gather(1, read_page.unsqueeze(1)).squeeze(1).to(torch.int64)
        valid = (pos > 0) & (read_phys > 0) & (read_phys < num_blocks)
        safe_phys = read_phys.clamp(min=0, max=num_blocks - 1)

        elem_size = torch.empty((), dtype=resolved_dtype).element_size()
        need_bytes = math.prod(target_shape) * elem_size
        base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
        if int(base_u8.shape[1]) < need_bytes:
            raise RuntimeError(
                f"Opaque pool row too small: {base_u8.shape[1]} bytes < {need_bytes} bytes "
                f"for tag={tag} layer={layer_idx} shape={target_shape} dtype={resolved_dtype}"
            )
        gathered = (
            base_u8[:, :need_bytes]
            .index_select(0, safe_phys)
            .view(resolved_dtype)
            .reshape(bsz, *target_shape)
            .contiguous()
        )
        valid_bc = valid.view(bsz, *([1] * len(target_shape)))
        return gathered.masked_fill_(~valid_bc, fill_value)

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

        new_last = seq_before_dev.to(torch.int64).clamp(min=0)
        write_page = (new_last // tpb).clamp(max=max_blocks - 1)
        write_phys = block_table.gather(1, write_page.unsqueeze(1)).squeeze(1).to(torch.int32)
        write_phys = torch.where(seq_before_dev >= 0, write_phys, 0)

        src_u8 = tensor.contiguous().reshape(bsz, -1).view(torch.uint8)
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

        block_sz = 1024
        grid = (bsz, int(triton.cdiv(n_elems, block_sz)))
        _write_opaque_rows_triton[grid](
            dst_view,
            src_view,
            write_phys,
            int(dst_view.stride(0)),
            int(src_view.stride(0)),
            n_elems,
            num_blocks,
            BLOCK_SIZE=block_sz,
        )

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

        emb = model.embed_tokens(input_ids_2d)
        hyper_hidden = emb.repeat(1, 1, self.cfg.hc_count)
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn: Dict[int, Any] = {}
        full_kv_layer_caches: Dict[int, Any] = {}
        source_idx_caches: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
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
                    ngram_context=self._read_opaque_batch_gpu(
                        ple_ctx_cache,
                        bt_ctx,
                        seq_lens_before_dev,
                        PLE_NGRAM_CTX_TAG,
                        lid,
                        fill_value=self.cfg.eos_token_id,
                    ),
                    conv_state=self._read_opaque_batch_gpu(
                        ple_st_cache,
                        bt_st,
                        seq_lens_before_dev,
                        PLE_STATE_TAG,
                        lid,
                        target_dtype=dtype,
                    ),
                )
                hyper_hidden = layer.ple(hyper_hidden, input_ids_2d, state=ple_state)
                self._write_opaque_batch_gpu(
                    ple_st_cache, bt_st, seq_lens_before_dev, ple_state.conv_state, PLE_STATE_TAG, lid
                )
                self._write_opaque_batch_gpu(
                    ple_ctx_cache,
                    bt_ctx,
                    seq_lens_before_dev,
                    ple_state.ngram_context,
                    PLE_NGRAM_CTX_TAG,
                    lid,
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
                deferred = None
            if aux_out is not None and lid in cap_slots:
                attn_flat = attn_in.reshape(bsz, self.cfg.hidden_size)
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * self.cfg.hidden_size : (slot_i + 1) * self.cfg.hidden_size].copy_(
                        attn_flat
                    )

            # 3. Attention (Linear GDN or Full MQA-1)
            if layer.layer_type == "linear_attention":
                assert layer.linear_attn is not None
                if layer.is_consumer:
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
                if not layer.is_consumer:
                    kv_c = layer_caches[FULL_ATTN_KV_TAG]
                    sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                    bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                    k_sc_st: Optional[ShortConvState] = None
                    v_sc_st: Optional[ShortConvState] = None
                    bt_sc = None
                    if sc_c is not None:
                        bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
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
                        q_idx, raw_k_idx = idx.project(attn_in)
                        use_pad_mask = decode_attn_handle.is_cuda_graph or _is_padded_group(
                            attention_inputs, FULL_ATTN_KV_TAG
                        )
                        active_rows = (
                            (bt_idx_st != 0).any(dim=1) if use_pad_mask else None
                        )
                        visible_lens = (
                            torch.where(active_rows, seq_lens_before_dev + 1, 0)
                            if active_rows is not None
                            else (seq_lens_before_dev + 1)
                        )
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

                    attn_out = layer.self_attn.forward_paged_decode(
                        attn_in,
                        positions_2d,
                        kv_c,
                        decode_fn,
                        block_table=bt_kv,
                        seq_before_dev=seq_lens_before_dev,
                        k_sconv_state=k_sc_st,
                        v_sconv_state=v_sc_st,
                        write_kv_fn=_store_decode_full_attn_kv_gpu,
                    )
                    full_kv_layer_caches[lid] = kv_c
                    if (
                        sc_c is not None
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
                        q_idx, _ = idx.project(attn_in)
                        use_pad_mask = decode_attn_handle.is_cuda_graph or _is_padded_group(
                            attention_inputs, FULL_ATTN_KV_TAG
                        )
                        active_rows = (
                            (src_bt_idx_kv != 0).any(dim=1) if use_pad_mask else None
                        )
                        visible_lens = (
                            torch.where(active_rows, seq_lens_before_dev + 1, 0)
                            if active_rows is not None
                            else (seq_lens_before_dev + 1)
                        )
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

                    attn_out = layer.self_attn.forward_paged_decode(
                        attn_in,
                        positions_2d,
                        src_kv_c,
                        c_decode_fn,
                    )

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
                    self._read_opaque_batch_gpu(
                        c_ao,
                        bt_ao,
                        seq_lens_before_dev,
                        tag_out,
                        lid,
                        target_dtype=dtype,
                    )
                )
                attn_out = layer.attn_output_sconv(attn_out, state=ao_state)
                self._write_opaque_batch_gpu(
                    c_ao,
                    bt_ao,
                    seq_lens_before_dev,
                    ao_state.conv_state,
                    tag_out,
                    lid,
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
            mlp_out = layer.mlp(mlp_in)

            # 6. FFN Output ShortConv + clamp
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                fo_state = ShortConvState(
                    self._read_opaque_batch_gpu(
                        c_fo,
                        bt_fo,
                        seq_lens_before_dev,
                        FFN_OUTPUT_SCONV_TAG,
                        lid,
                        target_dtype=dtype,
                    )
                )
                mlp_out = layer.ffn_output_sconv(mlp_out, state=fo_state)
                self._write_opaque_batch_gpu(
                    c_fo,
                    bt_fo,
                    seq_lens_before_dev,
                    fo_state.conv_state,
                    FFN_OUTPUT_SCONV_TAG,
                    lid,
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
                        q_idx, raw_k_idx = idx.project(attn_in)
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
                        )
                        source_idx_local[lid] = (idx_kv_pool, bt_idx_kv)

                    # NOTE: FullAttn MQA-1 needs exact per-request valid KV slice when batch has
                    # different sequence_lengths! When all seq_lens_before match, batched attention is exact.
                    if len(set(seq_lens_before)) == 1:
                        attn_out = layer.self_attn(
                            attn_in,
                            positions=positions_2d,
                            kv_cache=ephemeral_kv,
                            k_sconv_state=k_sc_st,
                            v_sconv_state=v_sc_st,
                            seq_offset=max_prev,
                            qsa_selected=src_selected,
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
                            out_b = layer.self_attn(
                                attn_in[b : b + 1],
                                positions=positions_2d[b : b + 1],
                                kv_cache=kv_b,
                                k_sconv_state=ks_b,
                                v_sconv_state=vs_b,
                                seq_offset=plen,
                                qsa_selected=sel_b,
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
                    if layer.self_attn.indexer is not None:
                        idx = layer.self_attn.indexer
                        src_idx_kv_pool, src_bt_idx_kv = source_idx_local[src_lid]
                        q_idx, _ = idx.project(attn_in)
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
                        )
                    shared_kv = full_kv_ephemeral[src_lid]
                    per_req = getattr(shared_kv, "_per_req", None)
                    if per_req is None:
                        attn_out = layer.self_attn(
                            attn_in,
                            positions=positions_2d,
                            kv_cache=shared_kv,
                            seq_offset=max_prev,
                            qsa_selected=cons_selected,
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
                            req_outs.append(
                                layer.self_attn(
                                    attn_in[b : b + 1],
                                    positions=positions_2d[b : b + 1],
                                    kv_cache=kv_b,
                                    seq_offset=plen,
                                    qsa_selected=sel_b,
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
            if hasattr(layer.mlp, "expert_bias") or hasattr(layer.mlp, "w13_u8"):
                mlp_out = layer.mlp(mlp_in)
            else:
                mlp_out = layer.mlp(mlp_in)

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
        flat_positions: torch.Tensor,
        input_lengths: List[int],
        prefix_lengths: List[int],
        kv_cache: Any,
        attention_inputs: Any,
        writers: Mapping[str, Any],
        *,
        enable_yoco_decode: Optional[bool] = None,
        capture_layer_ids: Sequence[int] = (),
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prefill path supporting ragged variable-length batching + prefix reuse + PD publication."""
        device = flat_input_ids.device
        dtype = model.embed_tokens.weight.dtype
        bsz = len(input_lengths)
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )

        offsets: List[int] = []
        cur = 0
        for ln in input_lengths:
            offsets.append(cur)
            cur += ln

        seg_tpb = 0
        if kv_cache is not None and not isinstance(kv_cache, LocalKVCache):
            for lid_probe in range(len(model.layers)):
                grps = kv_cache.get_layer_cache_groups(lid_probe)
                if grps:
                    seg_tpb = int(grps[0].seq_size_per_block)
                    break

        seg_specs: List[Tuple[int, int, int, int]] = []
        for b in range(bsz):
            t_in = input_lengths[b]
            t_pfx = prefix_lengths[b]
            if t_in == 0:
                continue
            if seg_tpb <= 0:
                seg_specs.append((b, offsets[b], t_in, t_pfx))
            else:
                pos_c = t_pfx
                rem = t_in
                off_c = offsets[b]
                while rem > 0:
                    next_bound = ((pos_c // seg_tpb) + 1) * seg_tpb
                    take = min(rem, next_bound - pos_c)
                    seg_specs.append((b, off_c, take, pos_c))
                    pos_c += take
                    off_c += take
                    rem -= take

        num_segs = len(seg_specs)
        total_tokens = int(flat_input_ids.numel())
        emb_flat = model.embed_tokens(flat_input_ids)
        hyper_reqs = [
            emb_flat[off_c : off_c + ln_c].unsqueeze(0).repeat(1, 1, self.cfg.hc_count)
            for _, off_c, ln_c, _ in seg_specs
        ]
        tokens_reqs = [
            flat_input_ids[off_c : off_c + ln_c].unsqueeze(0)
            for _, off_c, ln_c, _ in seg_specs
        ]
        pos_reqs = [
            flat_positions[off_c : off_c + ln_c].unsqueeze(0)
            for _, off_c, ln_c, _ in seg_specs
        ]
        prev_latents: List[Optional[torch.Tensor]] = [None] * num_segs
        deferreds: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_segs
        shared_gdn_reqs: List[Dict[int, Any]] = [{} for _ in range(num_segs)]
        source_prefix_ssm_reqs: List[Dict[int, Optional[torch.Tensor]]] = [{} for _ in range(num_segs)]
        full_kv_reqs: List[Dict[int, FullAttnKVCache]] = [{} for _ in range(num_segs)]
        full_qsa_raw_reqs: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_segs)
        ]
        source_idx_prefill: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None
        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = torch.empty(
                (total_tokens, len(capture_layer_ids) * self.cfg.hidden_size),
                dtype=dtype,
                device=device,
            )

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = (
                {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}
                if kv_cache is not None
                else {}
            )

            for s_idx, (bt_b, _seg_off, t_in, t_pfx) in enumerate(seg_specs):
                t_tot = t_pfx + t_in
                h_b = hyper_reqs[s_idx]
                tok_b = tokens_reqs[s_idx]
                pos_b = pos_reqs[s_idx]
                pl_b = prev_latents[s_idx]
                df_b = deferreds[s_idx]

                # 1. Optional PLE
                if layer.ple is not None:
                    if df_b is not None:
                        h_b = layer.attn_hyper_connection.combine(df_b[0], h_b, df_b[1])
                        df_b = None
                    if kv_cache is not None:
                        c_st = layer_caches[PLE_STATE_TAG]
                        c_ctx = layer_caches[PLE_NGRAM_CTX_TAG]
                        bt_st = _block_table_for_tag(attention_inputs, PLE_STATE_TAG)
                        bt_ctx = _block_table_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
                        ple_state = PLELayerState(
                            ngram_context=self._read_opaque_batch(
                                c_ctx,
                                bt_ctx[bt_b : bt_b + 1],
                                [t_pfx],
                                PLE_NGRAM_CTX_TAG,
                                lid,
                                fill_value=self.cfg.eos_token_id,
                            ),
                            conv_state=self._read_opaque_batch(
                                c_st,
                                bt_st[bt_b : bt_b + 1],
                                [t_pfx],
                                PLE_STATE_TAG,
                                lid,
                                target_dtype=dtype,
                            ),
                        )
                    else:
                        ple_state = layer.ple.alloc_state(1, device=device, dtype=dtype)
                    h_b = layer.ple(h_b, tok_b, state=ple_state)
                    if kv_cache is not None:
                        self._write_opaque_batch(
                            c_st, bt_st[bt_b : bt_b + 1], [t_tot], ple_state.conv_state, PLE_STATE_TAG, lid
                        )
                        self._write_opaque_batch(
                            c_ctx, bt_ctx[bt_b : bt_b + 1], [t_tot], ple_state.ngram_context, PLE_NGRAM_CTX_TAG, lid
                        )

                # 2. Attn HC
                if df_b is None:
                    attn_in, h_b, attn_inj, pl_b = layer.attn_hyper_connection.mix_with_latent(
                        h_b, pl_b
                    )
                else:
                    attn_in, h_b, attn_inj, pl_b = (
                        layer.attn_hyper_connection.combine_and_mix_with_latent(
                            df_b[0], h_b, df_b[1], pl_b
                        )
                    )
                assert attn_inj is not None
                if aux_out is not None and lid in cap_slots:
                    for slot_i in cap_slots[lid]:
                        aux_out[_seg_off : _seg_off + t_in, slot_i * self.cfg.hidden_size : (slot_i + 1) * self.cfg.hidden_size].copy_(
                            attn_in[0]
                        )

                # 3. Attention Core
                if layer.linear_attn is not None:
                    if layer.is_consumer:
                        q_state = None
                        bt_lin_c = None
                        base_lin_c = None
                        tpb_lin_c = 0
                        if LINEAR_ATTN_TAG in layer_caches and kv_cache is not None:
                            lin_cache = layer_caches[LINEAR_ATTN_TAG]
                            bt_lin_c = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                            tpb_lin_c = int(lin_cache.seq_size_per_block)
                            base_lin_c = lin_cache.kv_cache_base
                            if t_pfx > 0:
                                page = (t_pfx - 1) // tpb_lin_c
                                phys = _check_phys_block(
                                    int(bt_lin_c[bt_b, page].item()),
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
                                q_state = GDNLayerState(
                                    conv_state=conv_b.unsqueeze(0).clone(),
                                    ssm_state=ssm_b.unsqueeze(0).clone(),
                                )
                            else:
                                q_state = GDNLayerState(
                                    conv_state=torch.zeros(
                                        (1, *self.gdn_consumer_conv_shape), dtype=dtype, device=device
                                    ),
                                    ssm_state=torch.zeros(
                                        (1, *self.ssm_shape), dtype=torch.float32, device=device
                                    ),
                                )
                        elif GDN_SSM_STATE_TAG in layer_caches and kv_cache is not None:
                            c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                            bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                            ssm_b_t = self._read_opaque_batch(
                                c_ssm,
                                bt_ssm[bt_b : bt_b + 1],
                                [t_pfx],
                                GDN_SSM_STATE_TAG,
                                lid,
                                target_dtype=torch.float32,
                            )
                            if GDN_CONV_STATE_TAG in layer_caches:
                                c_q = layer_caches[GDN_CONV_STATE_TAG]
                                bt_q = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                                conv_b_t = self._read_opaque_batch(
                                    c_q,
                                    bt_q[bt_b : bt_b + 1],
                                    [t_pfx],
                                    GDN_CONSUMER_Q_CONV_TAG,
                                    lid,
                                    target_dtype=dtype,
                                )
                            else:
                                conv_b_t = torch.zeros(
                                    (1, *self.gdn_consumer_conv_shape), dtype=dtype, device=device
                                )
                            q_state = GDNLayerState(conv_state=conv_b_t, ssm_state=ssm_b_t)
                        elif GDN_CONSUMER_Q_CONV_TAG in layer_caches and kv_cache is not None:
                            c_q = layer_caches[GDN_CONSUMER_Q_CONV_TAG]
                            bt_q = _block_table_for_tag(attention_inputs, GDN_CONSUMER_Q_CONV_TAG)
                            ephemeral_ssm: Optional[torch.Tensor] = None
                            if yoco_on and t_pfx > 0 and t_in > 1:
                                src_pfx = source_prefix_ssm_reqs[s_idx].get(layer.source_layer_idx)  # type: ignore[arg-type]
                                if src_pfx is not None:
                                    ephemeral_ssm = src_pfx.clone()
                            elif not yoco_on:
                                ephemeral_ssm = torch.zeros(
                                    (1, *self.ssm_shape), dtype=torch.float32, device=device
                                )
                            q_state = GDNLayerState(
                                conv_state=self._read_opaque_batch(
                                    c_q,
                                    bt_q[bt_b : bt_b + 1],
                                    [t_pfx],
                                    GDN_CONSUMER_Q_CONV_TAG,
                                    lid,
                                    target_dtype=dtype,
                                ),
                                ssm_state=ephemeral_ssm,
                            )
                        elif not yoco_on:
                            q_state = layer.linear_attn.create_state(
                                1, device=device, dtype=dtype, allocate_consumer_ssm=True
                            )
                        attn_out, _ = layer.linear_attn(
                            attn_in,
                            state=q_state,
                            shared=shared_gdn_reqs[s_idx][layer.source_layer_idx],  # type: ignore[index]
                            enable_yoco_decode=yoco_on,
                        )
                        if q_state is not None and LINEAR_ATTN_TAG in layer_caches and kv_cache is not None and bt_lin_c is not None and base_lin_c is not None:
                            assert q_state.ssm_state is not None
                            end_page = (t_tot - 1) // tpb_lin_c
                            phys = _check_phys_block(
                                int(bt_lin_c[bt_b, end_page].item()),
                                int(base_lin_c.shape[0]),
                                LINEAR_ATTN_TAG,
                                lid,
                            )
                            _pack_linear_block(
                                base_lin_c[phys], q_state.ssm_state[0], q_state.conv_state[0]
                            )
                        elif q_state is not None and GDN_SSM_STATE_TAG in layer_caches and kv_cache is not None:
                            assert q_state.ssm_state is not None
                            self._write_opaque_batch(
                                c_ssm, bt_ssm[bt_b : bt_b + 1], [t_tot], q_state.ssm_state, GDN_SSM_STATE_TAG, lid
                            )
                            if GDN_CONV_STATE_TAG in layer_caches:
                                self._write_opaque_batch(
                                    c_q, bt_q[bt_b : bt_b + 1], [t_tot], q_state.conv_state, GDN_CONV_STATE_TAG, lid
                                )
                        elif q_state is not None and GDN_CONSUMER_Q_CONV_TAG in layer_caches and kv_cache is not None:
                            self._write_opaque_batch(
                                c_q, bt_q[bt_b : bt_b + 1], [t_tot], q_state.conv_state, GDN_CONSUMER_Q_CONV_TAG, lid
                            )
                    else:
                        use_split_gdn = kv_cache is not None and GDN_SSM_STATE_TAG in layer_caches
                        if use_split_gdn:
                            c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                            bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                            ssm_t = self._read_opaque_batch(
                                c_ssm,
                                bt_ssm[bt_b : bt_b + 1],
                                [t_pfx],
                                GDN_SSM_STATE_TAG,
                                lid,
                                target_dtype=torch.float32,
                            )
                            if GDN_CONV_STATE_TAG in layer_caches:
                                c_conv = layer_caches[GDN_CONV_STATE_TAG]
                                bt_conv = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                                conv_t = self._read_opaque_batch(
                                    c_conv,
                                    bt_conv[bt_b : bt_b + 1],
                                    [t_pfx],
                                    GDN_CONV_STATE_TAG,
                                    lid,
                                    target_dtype=dtype,
                                )
                            else:
                                c_conv = None
                                bt_conv = None
                                conv_t = torch.zeros((1, *self.gdn_conv_shape), dtype=dtype, device=device)
                            gdn_state = GDNLayerState(conv_state=conv_t, ssm_state=ssm_t)
                        elif kv_cache is not None:
                            lin_cache = layer_caches[LINEAR_ATTN_TAG]
                            bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                            tpb = int(lin_cache.seq_size_per_block)
                            base_lin = lin_cache.kv_cache_base
                            if t_pfx > 0:
                                page = (t_pfx - 1) // tpb
                                phys = _check_phys_block(int(bt_lin[bt_b, page].item()), int(base_lin.shape[0]), LINEAR_ATTN_TAG, lid)
                                ssm_b, conv_b = _unpack_linear_block(
                                    base_lin[phys], self.ssm_shape, self.gdn_conv_shape, dtype
                                )
                                gdn_state = GDNLayerState(
                                    conv_state=conv_b.unsqueeze(0).clone(),
                                    ssm_state=ssm_b.unsqueeze(0).clone(),
                                )
                            else:
                                gdn_state = GDNLayerState(
                                    conv_state=torch.zeros((1, *self.gdn_conv_shape), dtype=dtype, device=device),
                                    ssm_state=torch.zeros((1, *self.ssm_shape), dtype=torch.float32, device=device),
                                )
                        else:
                            gdn_state = layer.linear_attn.create_state(1, device=device, dtype=dtype)

                        if layer.is_source:
                            source_prefix_ssm_reqs[s_idx][lid] = (
                                gdn_state.ssm_state.clone()
                                if (t_pfx > 0 and gdn_state.ssm_state is not None)
                                else None
                            )
                        attn_out, published = layer.linear_attn(
                            attn_in,
                            state=gdn_state,
                            shared=None,
                            enable_yoco_decode=yoco_on,
                        )
                        if published is not None:
                            shared_gdn_reqs[s_idx][lid] = published
                        if use_split_gdn:
                            assert gdn_state.ssm_state is not None
                            self._write_opaque_batch(
                                c_ssm, bt_ssm[bt_b : bt_b + 1], [t_tot], gdn_state.ssm_state, GDN_SSM_STATE_TAG, lid
                            )
                            if c_conv is not None and bt_conv is not None:
                                self._write_opaque_batch(
                                    c_conv, bt_conv[bt_b : bt_b + 1], [t_tot], gdn_state.conv_state, GDN_CONV_STATE_TAG, lid
                                )
                        elif kv_cache is not None:
                            end_page = (t_tot - 1) // tpb
                            phys = _check_phys_block(int(bt_lin[bt_b, end_page].item()), int(base_lin.shape[0]), LINEAR_ATTN_TAG, lid)
                            _pack_linear_block(base_lin[phys], gdn_state.ssm_state[0], gdn_state.conv_state[0])
                else:
                    assert layer.self_attn is not None
                    if not layer.is_consumer:
                        if kv_cache is not None:
                            kv_c = layer_caches[FULL_ATTN_KV_TAG]
                            bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                            tpb = int(kv_c.seq_size_per_block)
                            k_buf = torch.zeros(
                                (1, t_tot, self.cfg.num_key_value_heads, self.cfg.head_dim),
                                dtype=dtype,
                                device=device,
                            )
                            v_buf = torch.zeros_like(k_buf)
                            if t_pfx > 0:
                                k_p, v_p = _load_single_req_full_attn_kv(
                                    kv_c.kv_cache_base,
                                    bt_kv[bt_b],
                                    t_pfx,
                                    tpb,
                                    num_kv_heads=self.cfg.num_key_value_heads,
                                    head_dim=self.cfg.head_dim,
                                )
                                k_buf[0, :t_pfx].copy_(k_p)
                                v_buf[0, :t_pfx].copy_(v_p)
                            kv_obj = FullAttnKVCache(k_cache=k_buf, v_cache=v_buf, seq_len=t_pfx)
                            sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                            k_sc_st = v_sc_st = None
                            if sc_c is not None:
                                bt_sc = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_SCONV_TAG)
                                kv_sc_buf = self._read_opaque_batch(
                                    sc_c,
                                    bt_sc[bt_b : bt_b + 1],
                                    [t_pfx],
                                    FULL_ATTN_KV_SCONV_TAG,
                                    lid,
                                    target_dtype=dtype,
                                )
                                k_sc_st = ShortConvState(kv_sc_buf[:, 0].clone())
                                v_sc_st = ShortConvState(kv_sc_buf[:, 1].clone())
                            pref_selected: Optional[torch.Tensor] = None
                            if layer.self_attn.indexer is not None:
                                idx = layer.self_attn.indexer
                                idx_kv_c = layer_caches[INDEXER_KV_TAG]
                                idx_st_c = layer_caches[INDEXER_STATE_TAG]
                                bt_idx_kv_full = _block_table_for_tag(attention_inputs, INDEXER_KV_TAG)
                                bt_idx_st_full = _block_table_for_tag(attention_inputs, INDEXER_STATE_TAG)
                                bt_idx_kv = bt_idx_kv_full[bt_b : bt_b + 1]
                                bt_idx_st = bt_idx_st_full[bt_b : bt_b + 1]
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
                                q_idx, raw_k_idx = idx.project(attn_in)
                                cu_1 = torch.tensor([0, t_in], dtype=torch.int32, device=device)
                                start_1 = torch.tensor([t_pfx], dtype=torch.int64, device=device)
                                write_qsa_prefill_cache(
                                    raw_k_idx.squeeze(0),
                                    cu_1,
                                    start_1,
                                    pos_b.squeeze(0),
                                    idx.k_norm_gamma,
                                    idx_kv_pool,
                                    bt_idx_kv,
                                    idx_st_pool,
                                    bt_idx_st,
                                    ratio=idx.compress_ratio,
                                    kv_tokens_per_block=idx_kv_tpb,
                                    state_tokens_per_block=idx_st_tpb,
                                    rotary_dim=layer.self_attn.indexer_rotary_dim,
                                    rope_theta=layer.self_attn.rope_theta,
                                    norm_eps=idx.norm_eps,
                                )
                                source_idx_prefill[lid] = (idx_kv_pool, bt_idx_kv_full)
                                if t_pfx == 0:
                                    cos_all, sin_all = build_qsa_neox_cos_sin(
                                        pos_b,
                                        layer.self_attn.indexer_rotary_dim,
                                        layer.self_attn.rope_theta,
                                        attn_in.dtype,
                                    )
                                    pref_selected = qsa_select_causal_from_projected(
                                        idx,
                                        q_idx,
                                        raw_k_idx,
                                        cos_all,
                                        sin_all,
                                    )
                                else:
                                    vis_q = torch.arange(
                                        t_pfx + 1,
                                        t_tot + 1,
                                        dtype=torch.int32,
                                        device=device,
                                    ).unsqueeze(0)
                                    max_ctx_p = max(1, t_tot // idx.compress_ratio)
                                    pref_selected = score_and_select_qsa_paged(
                                        q_idx,
                                        pos_b,
                                        idx_kv_pool,
                                        bt_idx_kv,
                                        vis_q,
                                        compress_ratio=idx.compress_ratio,
                                        token_budget=idx.token_budget,
                                        rotary_dim=layer.self_attn.indexer_rotary_dim,
                                        rope_theta=layer.self_attn.rope_theta,
                                        max_ctx_len=max_ctx_p,
                                    )

                            attn_out = layer.self_attn(
                                attn_in,
                                positions=pos_b,
                                kv_cache=kv_obj,
                                k_sconv_state=k_sc_st,
                                v_sconv_state=v_sc_st,
                                seq_offset=t_pfx,
                                qsa_selected=pref_selected,
                            )
                            _store_single_req_full_attn_kv(
                                kv_c.kv_cache_base,
                                bt_kv[bt_b],
                                kv_obj.k_cache[0, t_pfx:t_tot],
                                kv_obj.v_cache[0, t_pfx:t_tot],
                                t_pfx,
                                tpb,
                                num_kv_heads=self.cfg.num_key_value_heads,
                                head_dim=self.cfg.head_dim,
                            )
                            full_kv_reqs[s_idx][lid] = kv_obj
                            if sc_c is not None and k_sc_st is not None and v_sc_st is not None:
                                packed_sc = torch.stack([k_sc_st.conv_state, v_sc_st.conv_state], dim=1)
                                self._write_opaque_batch(
                                    sc_c, bt_sc[bt_b : bt_b + 1], [t_tot], packed_sc, FULL_ATTN_KV_SCONV_TAG, lid
                                )
                        else:
                            kv_obj = layer.self_attn.create_kv_cache(1, t_tot, device=device, dtype=dtype)
                            k_sc_st, v_sc_st = layer.self_attn.create_sconv_states(1, device=device, dtype=dtype)
                            qsa_nocache_kw: Dict[str, Any] = {}
                            if layer.self_attn.indexer is not None:
                                _src_kn = layer.self_attn.indexer.k_norm_gamma

                                def _rec_nocache(
                                    raw_k: torch.Tensor,
                                    _s_idx: int = s_idx,
                                    _lid: int = lid,
                                    _kn: torch.Tensor = _src_kn,
                                ) -> None:
                                    full_qsa_raw_reqs[_s_idx][_lid] = (raw_k, _kn)

                                qsa_nocache_kw["qsa_on_raw_keys"] = _rec_nocache
                            attn_out = layer.self_attn(
                                attn_in,
                                positions=pos_b,
                                kv_cache=kv_obj,
                                k_sconv_state=k_sc_st,
                                v_sconv_state=v_sc_st,
                                seq_offset=t_pfx,
                                **qsa_nocache_kw,
                            )
                            full_kv_reqs[s_idx][lid] = kv_obj
                    else:
                        cons_kw: Dict[str, Any] = {}
                        if layer.self_attn.indexer is not None:
                            idx = layer.self_attn.indexer
                            src_lid = layer.kv_cache_owner
                            if kv_cache is not None:
                                src_idx_kv_pool, src_bt_idx_kv_full = source_idx_prefill[src_lid]
                                q_idx, _ = idx.project(attn_in)
                                vis_q = torch.arange(
                                    t_pfx + 1,
                                    t_tot + 1,
                                    dtype=torch.int32,
                                    device=device,
                                ).unsqueeze(0)
                                max_ctx_p = max(1, t_tot // idx.compress_ratio)
                                cons_kw["qsa_selected"] = score_and_select_qsa_paged(
                                    q_idx,
                                    pos_b,
                                    src_idx_kv_pool,
                                    src_bt_idx_kv_full[bt_b : bt_b + 1],
                                    vis_q,
                                    compress_ratio=idx.compress_ratio,
                                    token_budget=idx.token_budget,
                                    rotary_dim=layer.self_attn.indexer_rotary_dim,
                                    rope_theta=layer.self_attn.rope_theta,
                                    max_ctx_len=max_ctx_p,
                                )
                            elif src_lid in full_qsa_raw_reqs[s_idx]:
                                src_raw, src_kn = full_qsa_raw_reqs[s_idx][src_lid]
                                cons_kw["qsa_source_raw_keys"] = src_raw
                                cons_kw["qsa_source_k_norm_gamma"] = src_kn
                        attn_out = layer.self_attn(
                            attn_in,
                            positions=pos_b,
                            kv_cache=full_kv_reqs[s_idx][layer.kv_cache_owner],
                            seq_offset=t_pfx,
                            **cons_kw,
                        )

                # 4. Attention Output ShortConv
                if layer.attn_output_sconv is not None:
                    tag_out = (
                        LINEAR_ATTN_OUTPUT_SCONV_TAG
                        if layer.layer_type == "linear_attention"
                        else FULL_ATTN_OUTPUT_SCONV_TAG
                    )
                    if kv_cache is not None:
                        c_ao = layer_caches[tag_out]
                        bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                        ao_state = ShortConvState(
                            self._read_opaque_batch(
                                c_ao,
                                bt_ao[bt_b : bt_b + 1],
                                [t_pfx],
                                tag_out,
                                lid,
                                target_dtype=dtype,
                            )
                        )
                    else:
                        ao_state = layer.attn_output_sconv.alloc_state(1, device=device, dtype=dtype)
                    attn_out = layer.attn_output_sconv(attn_out, state=ao_state)
                    if kv_cache is not None:
                        self._write_opaque_batch(
                            c_ao, bt_ao[bt_b : bt_b + 1], [t_tot], ao_state.conv_state, tag_out, lid
                        )
                if self.cfg.clip_attn_output is not None:
                    attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)

                # 5. MLP HC + MLP
                mlp_in, h_b, mlp_inj, pl_b = (
                    layer.mlp_hyper_connection.combine_and_mix_with_latent(
                        attn_out, h_b, attn_inj, pl_b
                    )
                )
                assert mlp_inj is not None
                mlp_out = layer.mlp(mlp_in)

                # 6. FFN Output ShortConv
                if layer.ffn_output_sconv is not None:
                    if kv_cache is not None:
                        c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                        bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                        fo_state = ShortConvState(
                            self._read_opaque_batch(
                                c_fo,
                                bt_fo[bt_b : bt_b + 1],
                                [t_pfx],
                                FFN_OUTPUT_SCONV_TAG,
                                lid,
                                target_dtype=dtype,
                            )
                        )
                    else:
                        fo_state = layer.ffn_output_sconv.alloc_state(1, device=device, dtype=dtype)
                    mlp_out = layer.ffn_output_sconv(mlp_out, state=fo_state)
                    if kv_cache is not None:
                        self._write_opaque_batch(
                            c_fo, bt_fo[bt_b : bt_b + 1], [t_tot], fo_state.conv_state, FFN_OUTPUT_SCONV_TAG, lid
                        )
                if self.cfg.clip_mlp_output is not None:
                    mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)

                hyper_reqs[s_idx] = h_b
                prev_latents[s_idx] = pl_b
                deferreds[s_idx] = (mlp_out, mlp_inj)

            # All batch rows have finished writing layer_idx physical pages -> publish to PD CacheStore!
            if writers:
                publish_qwen4_5_layer_caches(kv_cache, lid, writers)

        flat_out = torch.empty((total_tokens, self.cfg.hidden_size), dtype=dtype, device=device)
        for s_idx, (_bt_b, seg_off, t_in, _t_pfx) in enumerate(seg_specs):
            df_b = deferreds[s_idx]
            assert df_b is not None
            hid_b, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
                df_b[0], hyper_reqs[s_idx], df_b[1], prev_latent=None
            )
            flat_out[seg_off : seg_off + t_in].copy_(hid_b[0])
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
        """CUDA-graph-safe fixed-width [B, V] target verification with staged OPAQUE_STATE."""
        device = input_ids_2d.device
        dtype = model.embed_tokens.weight.dtype
        bsz, verify_w = int(input_ids_2d.shape[0]), int(input_ids_2d.shape[1])
        h_size = int(self.cfg.hidden_size)
        yoco_on = (
            self.cfg.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )

        tx_stage.begin_forward(seq_lens_before_dev, kv_cache, attention_inputs)

        cap_slots: Dict[int, List[int]] = {}
        aux_out: Optional[torch.Tensor] = None
        if capture_layer_ids:
            for slot_i, cap_lid in enumerate(capture_layer_ids):
                cap_slots.setdefault(int(cap_lid), []).append(slot_i)
            aux_out = tx_stage.get_or_alloc_aux(
                bsz * verify_w, len(capture_layer_ids) * h_size, dtype=dtype, device=device
            )

        emb = model.embed_tokens(input_ids_2d)  # [B, V, H]
        hyper_hidden = emb.repeat(1, 1, self.cfg.hc_count)
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn_steps: Dict[int, List[Any]] = {}
        full_kv_layer_caches: Dict[int, Any] = {}
        source_idx_caches: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for layer in model.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            layer_caches = {str(c.tag): c for c in kv_cache.get_layer_cache_groups(lid)}

            # 1. Optional PLE (step-by-step stateful scan; persistent pools untouched)
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
                    ngram_context=self._read_opaque_batch_gpu(
                        ple_ctx_cache,
                        bt_ctx,
                        seq_lens_before_dev,
                        PLE_NGRAM_CTX_TAG,
                        lid,
                        fill_value=self.cfg.eos_token_id,
                    ),
                    conv_state=self._read_opaque_batch_gpu(
                        ple_st_cache,
                        bt_st,
                        seq_lens_before_dev,
                        PLE_STATE_TAG,
                        lid,
                        target_dtype=dtype,
                    ),
                )
                stage_st = tx_stage.stage_tensor(
                    PLE_STATE_TAG, lid, ple_st_cache, bt_st, ple_state.conv_state.shape[1:], dtype
                )
                stage_ctx = tx_stage.stage_tensor(
                    PLE_NGRAM_CTX_TAG,
                    lid,
                    ple_ctx_cache,
                    bt_ctx,
                    ple_state.ngram_context.shape[1:],
                    torch.int64,
                )
                hyper_steps: List[torch.Tensor] = []
                for step in range(verify_w):
                    h_step = layer.ple(
                        hyper_hidden[:, step : step + 1],
                        input_ids_2d[:, step : step + 1],
                        state=ple_state,
                    )
                    stage_st[:, step].copy_(ple_state.conv_state)
                    stage_ctx[:, step].copy_(ple_state.ngram_context)
                    hyper_steps.append(h_step)
                hyper_hidden = torch.cat(hyper_steps, dim=1)

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
                attn_flat = attn_in.reshape(bsz * verify_w, h_size)
                for slot_i in cap_slots[lid]:
                    aux_out[:, slot_i * h_size : (slot_i + 1) * h_size].copy_(attn_flat)

            # 3. Attention (Linear GDN or Full MQA-1 / QSA) - step-by-step causal scan
            if layer.layer_type == "linear_attention":
                assert layer.linear_attn is not None
                attn_out_steps: List[torch.Tensor] = []
                if layer.is_consumer:
                    q_state: Optional[GDNLayerState] = None
                    stage_lin = None
                    stage_ssm = None
                    stage_conv = None
                    packed_template = None
                    ssm_n = math.prod(self.ssm_shape)
                    conv_n_c = math.prod(self.gdn_consumer_conv_shape)
                    if GDN_SSM_STATE_TAG in layer_caches:
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_buf = self._read_opaque_batch_gpu(
                            c_ssm, bt_ssm, seq_lens_before_dev, GDN_SSM_STATE_TAG, lid, target_dtype=torch.float32
                        )
                        stage_ssm = tx_stage.stage_tensor(
                            GDN_SSM_STATE_TAG, lid, c_ssm, bt_ssm, self.ssm_shape, torch.float32
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_q = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            q_conv_buf = self._read_opaque_batch_gpu(
                                c_conv, bt_q, seq_lens_before_dev, GDN_CONSUMER_Q_CONV_TAG, lid, target_dtype=dtype
                            )
                            stage_conv = tx_stage.stage_tensor(
                                GDN_CONV_STATE_TAG, lid, c_conv, bt_q, self.gdn_consumer_conv_shape, dtype
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
                            c_q, bt_q, seq_lens_before_dev, GDN_CONSUMER_Q_CONV_TAG, lid, target_dtype=dtype
                        )
                        stage_conv = tx_stage.stage_tensor(
                            GDN_CONSUMER_Q_CONV_TAG, lid, c_q, bt_q, self.gdn_consumer_conv_shape, dtype
                        )
                        q_state = GDNLayerState(
                            conv_state=q_conv_buf,
                            ssm_state=(
                                None
                                if yoco_on
                                else torch.zeros((bsz, *self.ssm_shape), dtype=torch.float32, device=device)
                            ),
                        )
                    elif LINEAR_ATTN_TAG in layer_caches:
                        c_lin = layer_caches[LINEAR_ATTN_TAG]
                        bt_q = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        packed = self._read_opaque_batch_gpu(
                            c_lin, bt_q, seq_lens_before_dev, LINEAR_ATTN_TAG, lid, target_dtype=torch.float32
                        )
                        ssm_buf = packed[:, :ssm_n].reshape(bsz, *self.ssm_shape).clone()
                        conv_buf = (
                            packed[:, ssm_n:]
                            .contiguous()
                            .view(torch.bfloat16)[:, :conv_n_c]
                            .reshape(bsz, *self.gdn_consumer_conv_shape)
                            .to(dtype)
                            .clone()
                        )
                        q_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)
                        packed_template = torch.zeros_like(packed)
                        stage_lin = tx_stage.stage_tensor(
                            LINEAR_ATTN_TAG, lid, c_lin, bt_q, packed.shape[1:], torch.float32
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported linear consumer cache layout at layer {lid}: {list(layer_caches.keys())}"
                        )

                    src_steps = shared_gdn_steps[layer.source_layer_idx]  # type: ignore[index]
                    for step in range(verify_w):
                        a_step, _ = layer.linear_attn(
                            attn_in[:, step : step + 1],
                            state=q_state,
                            shared=src_steps[step],
                            enable_yoco_decode=yoco_on,
                        )
                        attn_out_steps.append(a_step)
                        if stage_lin is not None and packed_template is not None and q_state is not None:
                            assert q_state.ssm_state is not None
                            packed_template[:, :ssm_n].copy_(q_state.ssm_state.reshape(bsz, ssm_n).float())
                            packed_template[:, ssm_n:].view(torch.bfloat16)[:, :conv_n_c].copy_(
                                q_state.conv_state.reshape(bsz, conv_n_c).to(torch.bfloat16)
                            )
                            stage_lin[:, step].copy_(packed_template)
                        else:
                            if stage_ssm is not None and q_state is not None and q_state.ssm_state is not None:
                                stage_ssm[:, step].copy_(q_state.ssm_state)
                            if stage_conv is not None and q_state is not None:
                                stage_conv[:, step].copy_(q_state.conv_state)
                else:
                    stage_lin = None
                    stage_ssm = None
                    stage_conv = None
                    packed_template = None
                    ssm_n = math.prod(self.ssm_shape)
                    conv_n = math.prod(self.gdn_conv_shape)
                    if LINEAR_ATTN_TAG in layer_caches:
                        c_lin = layer_caches[LINEAR_ATTN_TAG]
                        bt_lin = _block_table_for_tag(attention_inputs, LINEAR_ATTN_TAG)
                        packed = self._read_opaque_batch_gpu(
                            c_lin, bt_lin, seq_lens_before_dev, LINEAR_ATTN_TAG, lid, target_dtype=torch.float32
                        )
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
                        packed_template = torch.zeros_like(packed)
                        stage_lin = tx_stage.stage_tensor(
                            LINEAR_ATTN_TAG, lid, c_lin, bt_lin, packed.shape[1:], torch.float32
                        )
                    else:
                        if GDN_SSM_STATE_TAG not in layer_caches:
                            raise RuntimeError(
                                f"Unsupported linear cache layout at layer {lid}: {list(layer_caches.keys())}"
                            )
                        c_ssm = layer_caches[GDN_SSM_STATE_TAG]
                        bt_ssm = _block_table_for_tag(attention_inputs, GDN_SSM_STATE_TAG)
                        ssm_buf = self._read_opaque_batch_gpu(
                            c_ssm, bt_ssm, seq_lens_before_dev, GDN_SSM_STATE_TAG, lid, target_dtype=torch.float32
                        )
                        stage_ssm = tx_stage.stage_tensor(
                            GDN_SSM_STATE_TAG, lid, c_ssm, bt_ssm, self.ssm_shape, torch.float32
                        )
                        if GDN_CONV_STATE_TAG in layer_caches:
                            c_conv = layer_caches[GDN_CONV_STATE_TAG]
                            bt_conv = _block_table_for_tag(attention_inputs, GDN_CONV_STATE_TAG)
                            conv_buf = self._read_opaque_batch_gpu(
                                c_conv, bt_conv, seq_lens_before_dev, GDN_CONV_STATE_TAG, lid, target_dtype=dtype
                            )
                            stage_conv = tx_stage.stage_tensor(
                                GDN_CONV_STATE_TAG, lid, c_conv, bt_conv, self.gdn_conv_shape, dtype
                            )
                        else:
                            conv_buf = torch.zeros((bsz, *self.gdn_conv_shape), dtype=dtype, device=device)
                        gdn_state = GDNLayerState(conv_state=conv_buf, ssm_state=ssm_buf)

                    pub_steps: Optional[List[Any]] = [] if layer.linear_attn.publishes_shared else None
                    if stage_ssm is None and pub_steps is not None:
                        pub_ssm_stage = tx_stage.scratch_tensor(
                            ("pub_ssm", lid), (bsz, verify_w, *self.ssm_shape), torch.float32, device
                        )
                    else:
                        pub_ssm_stage = stage_ssm

                    for step in range(verify_w):
                        a_step, pub = layer.linear_attn(
                            attn_in[:, step : step + 1],
                            state=gdn_state,
                            shared=None,
                            enable_yoco_decode=yoco_on,
                        )
                        attn_out_steps.append(a_step)
                        assert gdn_state.ssm_state is not None
                        if pub_ssm_stage is not None:
                            pub_ssm_stage[:, step].copy_(gdn_state.ssm_state)
                        if pub is not None and pub_steps is not None:
                            pub_steps.append(
                                SharedGDNInputs(
                                    key=pub.key,
                                    value=pub.value,
                                    g=pub.g,
                                    beta=pub.beta,
                                    ssm_state=(
                                        pub_ssm_stage[:, step]
                                        if pub_ssm_stage is not None
                                        else None
                                    ),
                                )
                            )
                        if stage_lin is not None and packed_template is not None:
                            packed_template[:, :ssm_n].copy_(gdn_state.ssm_state.reshape(bsz, ssm_n).float())
                            packed_template[:, ssm_n:].view(torch.bfloat16)[:, :conv_n].copy_(
                                gdn_state.conv_state.reshape(bsz, conv_n).to(torch.bfloat16)
                            )
                            stage_lin[:, step].copy_(packed_template)
                        elif stage_conv is not None:
                            stage_conv[:, step].copy_(gdn_state.conv_state)

                    if pub_steps is not None:
                        shared_gdn_steps[lid] = pub_steps
                attn_out = torch.cat(attn_out_steps, dim=1)
            else:
                assert layer.self_attn is not None
                attn_out_steps = []
                if not layer.is_consumer:
                    kv_c = layer_caches[FULL_ATTN_KV_TAG]
                    sc_c = layer_caches.get(FULL_ATTN_KV_SCONV_TAG)
                    bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                    active_kv = (bt_kv != 0).any(dim=1)
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
                        init_idx_st = self._read_opaque_batch_gpu(
                            idx_st_c,
                            bt_idx_st,
                            seq_lens_before_dev,
                            INDEXER_STATE_TAG,
                            lid,
                            target_dtype=torch.float32,
                        )
                        stage_idx_st = tx_stage.stage_tensor(
                            INDEXER_STATE_TAG,
                            lid,
                            idx_st_c,
                            bt_idx_st,
                            init_idx_st.shape[1:],
                            torch.float32,
                        )
                        active_idx = (bt_idx_st != 0).any(dim=1)
                        idx_st_scratch = tx_stage.scratch_tensor(
                            ("idx_st_scratch", lid),
                            (bsz + 1, *init_idx_st.shape[1:]),
                            torch.float32,
                            device,
                        )
                        idx_st_scratch[0].zero_()
                        idx_st_scratch[1 : bsz + 1].copy_(init_idx_st)
                        scratch_bt = tx_stage.get_idx_scratch_bt(
                            lid, bsz, int(bt_idx_st.shape[1]), active_idx, device
                        )
                        source_idx_caches[lid] = (idx_kv_pool, bt_idx_kv)

                    for step in range(verify_w):
                        pos_step_dev = seq_lens_before_dev + step
                        pos_step_2d = positions_2d[:, step : step + 1]
                        step_in = attn_in[:, step : step + 1]
                        visible_lens = torch.where(active_kv, pos_step_dev + 1, 0)

                        if layer.self_attn.indexer is not None:
                            idx = layer.self_attn.indexer
                            q_idx, raw_k_idx = idx.project(step_in)
                            write_qsa_decode_cache_gpu_(
                                raw_k_idx.squeeze(1),
                                pos_step_dev,
                                idx.k_norm_gamma,
                                idx_kv_pool,
                                bt_idx_kv,
                                idx_st_scratch,
                                scratch_bt,
                                kv_tokens_per_block=idx_kv_tpb,
                                state_tokens_per_block=idx_st_tpb,
                                ratio=idx.compress_ratio,
                                rotary_dim=layer.self_attn.indexer_rotary_dim,
                                rope_theta=layer.self_attn.rope_theta,
                                norm_eps=idx.norm_eps,
                                valid=active_idx,
                            )
                            stage_idx_st[:, step].copy_(idx_st_scratch[1 : bsz + 1])
                            selected = score_and_select_qsa_paged(
                                q_idx,
                                pos_step_2d,
                                idx_kv_pool,
                                bt_idx_kv,
                                visible_lens,
                                compress_ratio=idx.compress_ratio,
                                token_budget=idx.token_budget,
                                rotary_dim=layer.self_attn.indexer_rotary_dim,
                                rope_theta=layer.self_attn.rope_theta,
                                max_ctx_len=None,
                                is_cuda_graph=verify_handle.is_cuda_graph,
                            )

                            def _qsa_step_fn(
                                q_3d: torch.Tensor,
                                kv_l: Any,
                                _bt: torch.Tensor = bt_kv,
                                _vl: torch.Tensor = visible_lens,
                                _sel: torch.Tensor = selected,
                            ) -> torch.Tensor:
                                return verify_handle.forward_qsa_step(
                                    q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, selected=_sel
                                )

                            step_fn = _qsa_step_fn
                        else:

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

                            step_fn = _dense_step_fn

                        a_step = layer.self_attn.forward_paged_decode(
                            step_in,
                            pos_step_2d,
                            kv_c,
                            step_fn,
                            block_table=bt_kv,
                            seq_before_dev=pos_step_dev,
                            k_sconv_state=k_sc_st,
                            v_sconv_state=v_sc_st,
                            write_kv_fn=_store_decode_full_attn_kv_gpu,
                        )
                        attn_out_steps.append(a_step)
                        if stage_kv_sc is not None and k_sc_st is not None and v_sc_st is not None:
                            stage_kv_sc[:, step, 0].copy_(k_sc_st.conv_state)
                            stage_kv_sc[:, step, 1].copy_(v_sc_st.conv_state)
                    full_kv_layer_caches[lid] = kv_c
                else:
                    src_kv_c = full_kv_layer_caches[layer.kv_cache_owner]
                    src_bt_kv = _block_table_for_tag(attention_inputs, FULL_ATTN_KV_TAG)
                    active_kv = (src_bt_kv != 0).any(dim=1)
                    for step in range(verify_w):
                        pos_step_dev = seq_lens_before_dev + step
                        pos_step_2d = positions_2d[:, step : step + 1]
                        step_in = attn_in[:, step : step + 1]
                        visible_lens = torch.where(active_kv, pos_step_dev + 1, 0)
                        if layer.self_attn.indexer is not None:
                            idx = layer.self_attn.indexer
                            src_idx_kv_pool, src_bt_idx_kv = source_idx_caches[layer.kv_cache_owner]
                            q_idx, _ = idx.project(step_in)
                            selected = score_and_select_qsa_paged(
                                q_idx,
                                pos_step_2d,
                                src_idx_kv_pool,
                                src_bt_idx_kv,
                                visible_lens,
                                compress_ratio=idx.compress_ratio,
                                token_budget=idx.token_budget,
                                rotary_dim=layer.self_attn.indexer_rotary_dim,
                                rope_theta=layer.self_attn.rope_theta,
                                max_ctx_len=None,
                                is_cuda_graph=verify_handle.is_cuda_graph,
                            )

                            def _qsa_cons_step_fn(
                                q_3d: torch.Tensor,
                                kv_l: Any,
                                _bt: torch.Tensor = src_bt_kv,
                                _vl: torch.Tensor = visible_lens,
                                _sel: torch.Tensor = selected,
                            ) -> torch.Tensor:
                                return verify_handle.forward_qsa_step(
                                    q_3d, kv_l, bt_kv=_bt, kv_lens=_vl, selected=_sel
                                )

                            c_step_fn = _qsa_cons_step_fn
                        else:

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

                            c_step_fn = _dense_cons_step_fn

                        a_step = layer.self_attn.forward_paged_decode(
                            step_in, pos_step_2d, src_kv_c, c_step_fn
                        )
                        attn_out_steps.append(a_step)
                attn_out = torch.cat(attn_out_steps, dim=1)

            # 4. Attention Output ShortConv (step-by-step)
            if layer.attn_output_sconv is not None:
                tag_out = (
                    LINEAR_ATTN_OUTPUT_SCONV_TAG
                    if layer.layer_type == "linear_attention"
                    else FULL_ATTN_OUTPUT_SCONV_TAG
                )
                c_ao = layer_caches[tag_out]
                bt_ao = _block_table_for_tag(attention_inputs, tag_out)
                ao_state = ShortConvState(
                    self._read_opaque_batch_gpu(
                        c_ao, bt_ao, seq_lens_before_dev, tag_out, lid, target_dtype=dtype
                    )
                )
                stage_ao = tx_stage.stage_tensor(
                    tag_out, lid, c_ao, bt_ao, ao_state.conv_state.shape[1:], dtype
                )
                ao_steps: List[torch.Tensor] = []
                for step in range(verify_w):
                    y_step = layer.attn_output_sconv(attn_out[:, step : step + 1], state=ao_state)
                    stage_ao[:, step].copy_(ao_state.conv_state)
                    ao_steps.append(y_step)
                attn_out = torch.cat(ao_steps, dim=1)
            if self.cfg.clip_attn_output is not None:
                attn_out = attn_out.clamp(-self.cfg.clip_attn_output, self.cfg.clip_attn_output)

            # 5. MLP Hyper-Connection + MoE/Dense MLP (1 batched pass across B*V tokens)
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                layer.mlp_hyper_connection.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
            assert mlp_inj is not None
            mlp_out = layer.mlp(mlp_in)

            # 6. FFN Output ShortConv (step-by-step)
            if layer.ffn_output_sconv is not None:
                c_fo = layer_caches[FFN_OUTPUT_SCONV_TAG]
                bt_fo = _block_table_for_tag(attention_inputs, FFN_OUTPUT_SCONV_TAG)
                fo_state = ShortConvState(
                    self._read_opaque_batch_gpu(
                        c_fo, bt_fo, seq_lens_before_dev, FFN_OUTPUT_SCONV_TAG, lid, target_dtype=dtype
                    )
                )
                stage_fo = tx_stage.stage_tensor(
                    FFN_OUTPUT_SCONV_TAG, lid, c_fo, bt_fo, fo_state.conv_state.shape[1:], dtype
                )
                fo_steps: List[torch.Tensor] = []
                for step in range(verify_w):
                    y_step = layer.ffn_output_sconv(mlp_out[:, step : step + 1], state=fo_state)
                    stage_fo[:, step].copy_(fo_state.conv_state)
                    fo_steps.append(y_step)
                mlp_out = torch.cat(fo_steps, dim=1)
            if self.cfg.clip_mlp_output is not None:
                mlp_out = mlp_out.clamp(-self.cfg.clip_mlp_output, self.cfg.clip_mlp_output)

            deferred = (mlp_out, mlp_inj)

        assert deferred is not None
        hidden, _, _, _ = model.hyper_connection_mixer.combine_and_mix_with_latent(
            deferred[0], hyper_hidden, deferred[1], prev_latent=None
        )
        tx_stage.mark_ready()
        hidden_out = hidden.reshape(bsz * verify_w, h_size)
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


@dataclass
class _TxUndoRecord:
    cache: Any
    tag: str
    layer_idx: int
    phys_ids: torch.Tensor
    commit_u8: torch.Tensor
    undo_u8: torch.Tensor


class _Qwen4_5OpaqueTxStage:
    """Static-address staging buffers for a fixed ``(B_cap, V)`` verify shape."""

    def __init__(self, bsz: int, verify_width: int) -> None:
        self.bsz = int(bsz)
        self.verify_width = int(verify_width)
        self.entries: Dict[Tuple[str, int], _TxStageEntry] = {}
        self.scratches: Dict[Any, torch.Tensor] = {}
        self._scratch_bts: Dict[int, torch.Tensor] = {}
        self.seq_before_buf: Optional[torch.Tensor] = None
        self.aux_buf: Optional[torch.Tensor] = None
        self.attention_inputs: Any = None
        self.kv_cache: Any = None
        self.ready = False

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
        self.seq_before_buf.copy_(seq_before_dev[: self.bsz].to(torch.int32))
        self.kv_cache = kv_cache
        self.attention_inputs = attention_inputs
        self.ready = False

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
        entry = self.entries.get(key)
        if (
            entry is None
            or tuple(entry.steps.shape) != wanted_shape
            or entry.steps.dtype != dtype
            or entry.steps.device != bt.device
        ):
            buf = torch.empty(wanted_shape, dtype=dtype, device=bt.device)
            entry = _TxStageEntry(
                tag=tag, layer_idx=int(layer_idx), cache=cache, bt=bt, steps=buf
            )
            self.entries[key] = entry
        else:
            entry.cache = cache
            entry.bt = bt
        return entry.steps

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


class _Qwen4_5OpaqueTxManager:
    """Two-phase commit + rollback manager for speculative ``OPAQUE_STATE`` pools."""

    def __init__(self, bridge: Qwen4_5PagedStateBridge, arch_cfg: Qwen4_5ArchConfig) -> None:
        self.bridge = bridge
        self.arch_cfg = arch_cfg
        self._graph_stages: Dict[Tuple[int, int], _Qwen4_5OpaqueTxStage] = {}
        self._eager_stage: Optional[_Qwen4_5OpaqueTxStage] = None
        self._active_pending: Optional[_Qwen4_5OpaqueTxStage] = None
        self._undo_records: List[_TxUndoRecord] = []
        self._committed = False

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
            entry.bt = _block_table_for_tag(attention_inputs, entry.tag)
        stage.ready = True
        self._active_pending = stage

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
        a_len = accept_len.to(device=dev, dtype=torch.int64)
        pos_before = stage.seq_before_buf[:live_bs].to(torch.int64)
        valid = a_len > 0
        v_w = stage.verify_width
        step_final = (a_len - 1).clamp(min=0, max=max(0, v_w - 1))
        rows_ar = torch.arange(live_bs, dtype=torch.int64, device=dev)

        undo_records: List[_TxUndoRecord] = []
        for entry in stage.entries.values():
            base = entry.cache.kv_cache_base
            num_blocks = int(base.shape[0])
            tpb = max(1, int(entry.cache.seq_size_per_block))
            bt_live = entry.bt[:live_bs]
            max_blocks = int(bt_live.shape[1]) if bt_live.ndim >= 2 else 0
            if num_blocks <= 0 or max_blocks <= 0:
                continue

            active_bt = (bt_live != 0).any(dim=1) & valid
            final_pos = (pos_before + a_len - 1).clamp(min=0)
            final_page = (final_pos // tpb).clamp(max=max_blocks - 1)
            final_phys = bt_live.gather(1, final_page.unsqueeze(1)).squeeze(1).to(torch.int32)
            final_phys = torch.where(
                active_bt & (final_phys > 0) & (final_phys < num_blocks), final_phys, 0
            )
            final_src = entry.steps[rows_ar, step_final]

            # Boundary preservation: if accept_len crossed a block boundary, also commit
            # the state at the last token of pos_before's block for prefix reuse.
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
            bnd_src = entry.steps[rows_ar, boundary_step.clamp(min=0, max=max(0, v_w - 1))]

            phys_ids = torch.cat([final_phys, bnd_phys], dim=0)
            src_states = torch.cat([final_src, bnd_src], dim=0)
            src_u8 = src_states.contiguous().reshape(2 * live_bs, -1).view(torch.uint8)
            need_bytes = int(src_u8.shape[1])
            base_u8 = base.reshape(num_blocks, -1).view(torch.uint8)
            if int(base_u8.shape[1]) < need_bytes:
                raise RuntimeError(
                    f"Opaque pool row too small during commit: {base_u8.shape[1]} < {need_bytes} "
                    f"for tag={entry.tag} layer={entry.layer_idx}"
                )
            safe_phys = phys_ids.to(torch.int64).clamp(min=0, max=num_blocks - 1)
            undo_u8 = base_u8[:, :need_bytes].index_select(0, safe_phys).contiguous()
            undo_records.append(
                _TxUndoRecord(
                    cache=entry.cache,
                    tag=entry.tag,
                    layer_idx=entry.layer_idx,
                    phys_ids=phys_ids,
                    commit_u8=src_u8,
                    undo_u8=undo_u8,
                )
            )

        self._undo_records = undo_records
        self._committed = False

    def finish_commit(self, commit: bool) -> None:
        if commit:
            for rec in self._undo_records:
                _write_opaque_rows_raw_gpu(
                    rec.cache.kv_cache_base, rec.phys_ids, rec.commit_u8, rec.tag, rec.layer_idx
                )
            self._committed = True
        else:
            if self._committed:
                for rec in self._undo_records:
                    _write_opaque_rows_raw_gpu(
                        rec.cache.kv_cache_base, rec.phys_ids, rec.undo_u8, rec.tag, rec.layer_idx
                    )
            self._committed = False
            self._undo_records.clear()
            self._active_pending = None

    def finalize_commit(self) -> None:
        self._undo_records.clear()
        self._committed = False
        self._active_pending = None


# ---------------------------------------------------------------------------
# Qwen4_5RTPModel (GptModelBase Integration)
# ---------------------------------------------------------------------------


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
        self.bridge = Qwen4_5PagedStateBridge(self.arch_config)
        self.tx_manager = _Qwen4_5OpaqueTxManager(self.bridge, self.arch_config)
        self._cpu_target_status = None
        tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1) if parallelism_config is not None else 1
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

        if backbone is not None:
            self.model = backbone
        else:
            dev = device or (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model = Qwen4_5Model(
                self.arch_config,
                dtype=dtype,
                device=dev,
                ep_size=ep_size,
                ep_rank=ep_rank,
                ep_group=ep_group,
            )
            if weight is not None:
                bind_qwen4_5_model_weights(
                    self.model,
                    weight,
                    self.arch_config,
                    lm_head=self.lm_head,
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    ep_group=ep_group,
                )

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
        if attn_inputs is None:
            return None
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        is_target_verify = bool(getattr(primary, "is_target_verify", False))
        if not is_target_verify and (
            isinstance(self.kv_cache, LocalKVCache) and not self.arch_config.has_qsa
        ):
            return None
        if not is_target_verify and bool(getattr(primary, "is_prefill", False)):
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
        attn_cfg.head_num = int(self.arch_config.num_attention_heads)
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
        return pfx[:batch_size].to(device=device, dtype=torch.int32)

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
        if attn_inputs is None:
            raise RuntimeError("PyModelInputs.attention_inputs is required")

        # Pick primary attention group (prefer "full" if present)
        primary = _select_group_for_tag(attn_inputs, FULL_ATTN_KV_TAG)
        is_target_verify = bool(getattr(primary, "is_target_verify", False))
        is_prefill = bool(getattr(primary, "is_prefill", False)) and not is_target_verify
        flat_ids = inputs.input_ids.reshape(-1)
        device = flat_ids.device
        cap_ids = self._mtp_aux_capture_layer_ids

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

        writers = (
            prepare_qwen4_5_cache_store_writers(
                attn_inputs, self.kv_cache, layer_count=len(self.model.layers)
            )
            if is_prefill and self.kv_cache is not None
            else {}
        )

        dev_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with dev_ctx:
            if is_target_verify:
                if not _HAS_TRITON or device.type != "cuda" or self.kv_cache is None:
                    raise RuntimeError(
                        "Qwen4_5 target verification requires CUDA + Triton + KVCache"
                    )
                raw_pfx_dev = getattr(primary, "prefix_lengths_device", None)
                pfx_dev = raw_pfx_dev() if callable(raw_pfx_dev) else raw_pfx_dev
                if not isinstance(pfx_dev, torch.Tensor) or pfx_dev.numel() == 0:
                    pfx_dev = getattr(primary, "prefix_lengths", None)
                if not isinstance(pfx_dev, torch.Tensor) or pfx_dev.numel() == 0:
                    raise RuntimeError("Target verification requires prefix_lengths")
                bsz = int(pfx_dev.numel())
                total_toks = int(flat_ids.numel())
                if bsz <= 0 or total_toks % bsz != 0:
                    raise ValueError(
                        f"Target verify total tokens {total_toks} not divisible by batch_size {bsz}"
                    )
                verify_w = total_toks // bsz
                seq_before_dev = pfx_dev.to(device=device, dtype=torch.int32)
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
                in_lens = [int(x) for x in primary.input_lengths.tolist()]
                pfx_lens = (
                    [int(x) for x in primary.prefix_lengths.tolist()]
                    if getattr(primary, "prefix_lengths", None) is not None
                    and primary.prefix_lengths.numel() == len(in_lens)
                    else [0] * len(in_lens)
                )
                flat_pos = self._build_local_positions(
                    primary, in_lens, pfx_lens, pfx_lens, True, device, inputs
                )
                res = self.bridge.run_prefill_ragged(
                    self.model,
                    flat_ids,
                    flat_pos,
                    in_lens,
                    pfx_lens,
                    self.kv_cache,
                    attn_inputs,
                    writers,
                    enable_yoco_decode=enable_yoco_decode,
                    capture_layer_ids=cap_ids,
                )
                if isinstance(res, tuple):
                    return self._wrap_outputs(res[0], res[1])
                return self._wrap_outputs(res)

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
                    seq_before_dev = primary.sequence_lengths.to(
                        device=device, dtype=torch.int32
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
            return self.lm_head(hidden_states)


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


def pack_padded_block_tensor(tables: List[List[int]], device: torch.device) -> torch.Tensor:
    max_cols = max((len(r) for r in tables), default=1)
    out = torch.zeros((len(tables), max_cols), dtype=torch.int32, device=device)
    for b, row in enumerate(tables):
        if row:
            out[b, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return out


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
    in_len_t = torch.tensor(list(input_lengths), dtype=torch.int32, device="cpu")
    seq_len_t = torch.tensor(list(sequence_lengths), dtype=torch.int32, device="cpu")
    pfx_len_t = torch.tensor(list(prefix_lengths), dtype=torch.int32, device="cpu")
    total_tok = int(sum(input_lengths))

    groups: Dict[str, LocalPyAttentionInputs] = {}
    for tag, tbl in tables_by_tag.items():
        tbl_cpu = pack_padded_block_tensor(tbl, torch.device("cpu"))
        tbl_dev = tbl_cpu.to(device)
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
