"""Qwen 4.5-Flash hybrid + YoCo + OPAQUE_STATE KV cache planner and spec builder.

Models the exact per-layer memory layout of Qwen 4.5-Flash:
  - 5 producer FullAttn layers (3, 7, 11, 15, 19): MQA-1 KV cache + KV short-conv + output short-conv
  - 5 consumer FullAttn layers (23, 27, 31, 35, 39): zero new KV cache (share layer 19) + output short-conv
  - 16 normal/source GDN layers: LINEAR conv(12288, 3) + FP32 SSM [128, 64, 64] + output short-conv
  - 14 consumer GDN layers: Q-conv(2048, 3) + zero SSM state in YoCo decode mode (share layer 20) + output short-conv
  - Layer 1 PLE: ``ple_conv_state`` [20480, 3] + ``ple_ngram_ctx`` [2] int64
  - All 40 layers: ``ffn_output_sconv`` [5120, 3]

Works both in pure Python (via ``Qwen4_5KVCachePlan`` / ``build_qwen4_5_kv_cache_plan``)
and inside the compiled RTP-LLM engine (via ``build_qwen4_5_kv_cache_spec_descs``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rtp_llm.config.qwen4_5_config import validate_qwen4_5_arch_config
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

try:
    from rtp_llm.ops import (
        CacheCapacityPolicyDesc,
        CacheGroupType,
        CacheReusePolicyDesc,
        CacheTailPolicyDesc,
        DataType,
        KVCacheSpecDesc,
        KVCacheSpecType,
        OpaqueBlockEntryCountMode,
        RoleType,
        TpCacheTransferMode,
    )

    from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
        INDEXER_KV_TAG,
        INDEXER_STATE_TAG,
        indexer_kv_desc,
        indexer_state_desc,
    )

    _HAS_RTP_OPS = True
except Exception:  # pragma: no cover - pure-Python fallback without C++ ops
    _HAS_RTP_OPS = False
    RoleType = None  # type: ignore[assignment,misc]
    INDEXER_KV_TAG = "indexer_kv"
    INDEXER_STATE_TAG = "indexer_state"
    indexer_kv_desc = None
    indexer_state_desc = None

DEFAULT_DECODE_OPAQUE_POOL_BLOCKS = 256
DEFAULT_DECODE_SPEC_LINEAR_POOL_BLOCKS = 512
DECODE_STATE_ROLLING_ENV = "QWEN4_5_DECODE_STATE_ROLLING"
DECODE_STATE_GROUP_TYPE_ENV = "QWEN4_5_DECODE_STATE_GROUP_TYPE"
DECODE_STATE_POOL_BLOCKS_ENV = "QWEN4_5_DECODE_STATE_POOL_BLOCKS"
DECODE_STATE_ACTIVE_TAIL_BLOCKS_ENV = "QWEN4_5_DECODE_STATE_ACTIVE_TAIL_BLOCKS"

FULL_ATTN_KV_TAG = "full"
LINEAR_ATTN_TAG = "linear"
GDN_SSM_STATE_TAG = "gdn_ssm_state"
GDN_CONV_STATE_TAG = "gdn_conv_state"
GDN_CONSUMER_Q_CONV_TAG = "gdn_consumer_q_conv"
FULL_ATTN_KV_SCONV_TAG = "full_attn_kv_sconv"
FULL_ATTN_K_SCONV_TAG = "full_attn_k_sconv"
FULL_ATTN_V_SCONV_TAG = "full_attn_v_sconv"
FULL_ATTN_OUTPUT_SCONV_TAG = "full_attn_output_sconv"
LINEAR_ATTN_OUTPUT_SCONV_TAG = "linear_attn_output_sconv"
FFN_OUTPUT_SCONV_TAG = "ffn_output_sconv"
PLE_STATE_TAG = "ple_conv_state"
PLE_NGRAM_CTX_TAG = "ple_ngram_ctx"

_KV_DTYPE_BYTES: Dict[str, int] = {
    "BF16": 2,
    "FP16": 2,
    "F16": 2,
    "FP8": 1,
    "F8_E4M3": 1,
    "FP8_E4M3": 1,
    "F32": 4,
    "FP32": 4,
}

_STATE_DTYPE_BYTES: Dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "FP16": 2,
    "F32": 4,
    "I64": 8,
}


@dataclass(frozen=True)
class OpaqueStateRegionPlan:
    """Per-layer fixed-size ``OPAQUE_STATE`` cache region descriptor."""

    tag: str
    dtype: str
    shape: Tuple[int, ...]
    bytes_per_req: int
    is_short_conv: bool = False
    conv_tensor_count: int = 1


@dataclass(frozen=True)
class Qwen4_5LayerCachePlan:
    """Complete KV/SSM/ShortConv/PLE cache plan for one decoder layer."""

    layer_idx: int
    layer_type: str
    is_sharing_consumer: bool
    sharing_source_layer: Optional[int]

    # Full Attention KV cache
    allocates_kv_cache: bool
    kv_source_layer: Optional[int]
    num_kv_heads: int
    head_dim: int
    kv_bytes_per_token: int

    # Linear Attention (GDN) LINEAR conv + SSM state
    gdn_conv_channels: int
    gdn_conv_state_len: int
    gdn_conv_bytes_per_req: int
    allocates_ssm_state: bool
    ssm_source_layer: Optional[int]
    ssm_state_shape: Optional[Tuple[int, int, int]]
    ssm_bytes_per_req: int

    # FullAttn producer KV short-conv metadata (matches Qwen4_5RuntimeState.k_sconv / v_sconv)
    has_kv_sconv: bool
    k_sconv_shape: Optional[Tuple[int, int]]
    v_sconv_shape: Optional[Tuple[int, int]]

    # PLE layer state metadata
    has_ple: bool
    ple_conv_state_shape: Optional[Tuple[int, int]]
    ple_ngram_ctx_shape: Optional[Tuple[int]]

    # OPAQUE_STATE regions attached to this layer
    opaque_regions: Tuple[OpaqueStateRegionPlan, ...]

    # QSA indexer KV + ring state (source non-GDN layers only)
    has_qsa_indexer_cache: bool = False
    indexer_kv_source_layer: Optional[int] = None
    indexer_head_dim: int = 0
    indexer_compress_ratio: int = 0

    @property
    def has_gdn_conv(self) -> bool:
        return self.gdn_conv_channels > 0 and self.gdn_conv_state_len > 0

    @property
    def short_conv_state_count(self) -> int:
        """Number of logical ShortConv state slots on this layer (kv_sconv paired = 1)."""
        count = 1 if self.has_gdn_conv else 0
        count += sum(1 for reg in self.opaque_regions if reg.is_short_conv)
        return count

    @property
    def short_conv_tensor_count(self) -> int:
        """Number of individual ShortConv tensors on this layer (k_sconv & v_sconv separate)."""
        count = 1 if self.has_gdn_conv else 0
        count += sum(
            reg.conv_tensor_count for reg in self.opaque_regions if reg.is_short_conv
        )
        return count

    @property
    def short_conv_bytes_per_req(self) -> int:
        return self.gdn_conv_bytes_per_req + sum(
            reg.bytes_per_req for reg in self.opaque_regions if reg.is_short_conv
        )

    @property
    def ple_ngram_ctx_bytes_per_req(self) -> int:
        return sum(
            reg.bytes_per_req
            for reg in self.opaque_regions
            if reg.tag == PLE_NGRAM_CTX_TAG
        )

    @property
    def fixed_state_bytes_per_req(self) -> int:
        return (
            self.ssm_bytes_per_req
            + self.short_conv_bytes_per_req
            + self.ple_ngram_ctx_bytes_per_req
        )


@dataclass(frozen=True)
class Qwen4_5KVCachePlan:
    """Whole-model KV cache, GDN SSM, ShortConv, and PLE state plan for Qwen 4.5-Flash."""

    arch_config: Qwen4_5ArchConfig
    enable_yoco_decode: bool
    kv_dtype: str
    kv_dtype_bytes: int
    layers: Tuple[Qwen4_5LayerCachePlan, ...]
    attention_sharing: Dict[int, int]

    @property
    def full_attn_kv_layer_indices(self) -> Tuple[int, ...]:
        return tuple(
            lp.layer_idx for lp in self.layers if lp.allocates_kv_cache
        )

    @property
    def full_attn_shared_consumer_indices(self) -> Tuple[int, ...]:
        return tuple(
            lp.layer_idx
            for lp in self.layers
            if lp.layer_type == "full_attention" and lp.is_sharing_consumer
        )

    @property
    def gdn_ssm_layer_indices(self) -> Tuple[int, ...]:
        return tuple(
            lp.layer_idx for lp in self.layers if lp.allocates_ssm_state
        )

    @property
    def gdn_shared_consumer_indices(self) -> Tuple[int, ...]:
        return tuple(
            lp.layer_idx
            for lp in self.layers
            if lp.layer_type == "linear_attention" and lp.is_sharing_consumer
        )

    @property
    def ple_layer_indices(self) -> Tuple[int, ...]:
        return tuple(lp.layer_idx for lp in self.layers if lp.has_ple)

    @property
    def num_full_attn_kv_layers(self) -> int:
        return len(self.full_attn_kv_layer_indices)

    @property
    def kv_bytes_per_token(self) -> int:
        return sum(lp.kv_bytes_per_token for lp in self.layers)

    @property
    def kv_kib_per_token(self) -> float:
        return self.kv_bytes_per_token / 1024.0

    @property
    def num_gdn_ssm_layers(self) -> int:
        return len(self.gdn_ssm_layer_indices)

    @property
    def ssm_bytes_per_req(self) -> int:
        return sum(lp.ssm_bytes_per_req for lp in self.layers)

    @property
    def ssm_mib_per_req(self) -> float:
        return self.ssm_bytes_per_req / float(1024**2)

    @property
    def num_short_conv_states(self) -> int:
        """Total logical ShortConv state slots across all layers (116 for 1109)."""
        return sum(lp.short_conv_state_count for lp in self.layers)

    @property
    def num_short_conv_tensors(self) -> int:
        """Total individual ShortConv tensors across all layers (121 for 1109)."""
        return sum(lp.short_conv_tensor_count for lp in self.layers)

    @property
    def short_conv_bytes_per_req(self) -> int:
        return sum(lp.short_conv_bytes_per_req for lp in self.layers)

    @property
    def short_conv_mib_per_req(self) -> float:
        return self.short_conv_bytes_per_req / float(1024**2)

    @property
    def ple_ngram_ctx_bytes_per_req(self) -> int:
        return sum(lp.ple_ngram_ctx_bytes_per_req for lp in self.layers)

    @property
    def total_fixed_state_bytes_per_req(self) -> int:
        return (
            self.ssm_bytes_per_req
            + self.short_conv_bytes_per_req
            + self.ple_ngram_ctx_bytes_per_req
        )

    @property
    def opaque_state_bytes_per_block(self) -> int:
        return self.total_fixed_state_bytes_per_req

    def estimate_capacity(
        self,
        total_budget_bytes: int,
        *,
        tokens_per_block: int = 128,
        is_decode_rolling: bool = False,
        opaque_pool_blocks: int = DEFAULT_DECODE_OPAQUE_POOL_BLOCKS,
    ) -> Dict[str, Any]:
        """Estimate FullAttn KV block capacity with or without Decode rolling opaque pool."""
        if total_budget_bytes < 0 or tokens_per_block <= 0 or opaque_pool_blocks < 0:
            raise ValueError(
                "total_budget_bytes and opaque_pool_blocks must be >= 0, tokens_per_block > 0"
            )
        per_block_full_kv_bytes = int(self.kv_bytes_per_token) * int(tokens_per_block)
        per_block_opaque_bytes = int(self.opaque_state_bytes_per_block)
        if is_decode_rolling:
            opaque_pool_bytes = int(opaque_pool_blocks) * per_block_opaque_bytes
            full_budget_bytes = max(0, int(total_budget_bytes) - opaque_pool_bytes)
            full_kv_blocks = (
                full_budget_bytes // per_block_full_kv_bytes
                if per_block_full_kv_bytes > 0
                else 0
            )
        else:
            combined = per_block_full_kv_bytes + per_block_opaque_bytes
            full_kv_blocks = int(total_budget_bytes) // combined if combined > 0 else 0
            opaque_pool_blocks = full_kv_blocks
            opaque_pool_bytes = full_kv_blocks * per_block_opaque_bytes
        return {
            "full_kv_blocks": full_kv_blocks,
            "full_kv_tokens": full_kv_blocks * int(tokens_per_block),
            "opaque_pool_blocks": int(opaque_pool_blocks),
            "opaque_pool_bytes": opaque_pool_bytes,
            "per_block_full_kv_bytes": per_block_full_kv_bytes,
            "per_block_opaque_bytes": per_block_opaque_bytes,
        }

    def total_bytes_for_seq_len(self, seq_len: int, batch_size: int = 1) -> int:
        if seq_len < 0 or batch_size < 0:
            raise ValueError("seq_len and batch_size must be non-negative")
        return batch_size * (
            self.kv_bytes_per_token * seq_len + self.total_fixed_state_bytes_per_req
        )


def resolve_kv_cache_tp_size(tp_size: int = 1, parallel_mode: Optional[str] = None) -> int:
    if parallel_mode:
        pm = str(parallel_mode).strip().lower()
        if pm == "dp4_ep4":
            return 1
        if pm in ("dp2_tp2_ep4", "dp2_tpsp2_ep4"):
            return 2 if int(tp_size or 1) <= 1 else int(tp_size)
        if pm in ("tp4_ep4", "tpsp4", "tpsp4_ep4", "sp4_tp4_ep4"):
            return 4 if int(tp_size or 1) <= 1 else int(tp_size)
    return max(1, int(tp_size or 1))


def build_qwen4_5_kv_cache_plan(
    arch_cfg: Qwen4_5ArchConfig,
    *,
    tp_size: int = 1,
    parallel_mode: Optional[str] = None,
    enable_yoco_decode: Optional[bool] = None,
    kv_dtype: str = "BF16",
    state_dtype: str = "BF16",
) -> Qwen4_5KVCachePlan:
    """Build the exact pure-Python KV/SSM/ShortConv/PLE cache plan for Qwen 4.5-Flash.

    When ``tp_size > 1`` (with matching ``P_tp == D_tp`` RDMA rank pairing),
    GDN SSM state and conv1d states are sharded per TP rank across heads
    (e.g. 32K/128V -> 8K/32V/rank at TP=4), while replicated MQA-1 KV/indexer
    and full-hidden output sconv/PLE states keep their full per-rank shape.
    """
    share_map = validate_qwen4_5_arch_config(arch_cfg)
    tp = resolve_kv_cache_tp_size(tp_size, parallel_mode)
    if tp > 1:
        if arch_cfg.linear_num_key_heads % tp != 0:
            raise ValueError(
                f"linear_num_key_heads ({arch_cfg.linear_num_key_heads}) must be divisible by tp_size ({tp})"
            )
        if arch_cfg.linear_num_value_heads % tp != 0:
            raise ValueError(
                f"linear_num_value_heads ({arch_cfg.linear_num_value_heads}) must be divisible by tp_size ({tp})"
            )
    use_yoco = (
        arch_cfg.enable_yoco_decode
        if enable_yoco_decode is None
        else bool(enable_yoco_decode)
    )
    kv_dt = str(kv_dtype or "BF16").strip().upper()
    if kv_dt not in _KV_DTYPE_BYTES:
        raise ValueError(f"Unsupported kv_dtype={kv_dtype!r}")
    kv_elem_bytes = _KV_DTYPE_BYTES[kv_dt]

    st_dt = state_dtype.upper()
    if st_dt not in _STATE_DTYPE_BYTES:
        raise ValueError(f"Unsupported state_dtype={state_dtype!r}")
    st_elem_bytes = _STATE_DTYPE_BYTES[st_dt]

    h = arch_cfg.hidden_size
    h_hc = arch_cfg.hc_count * h
    k_conv_state = max(0, arch_cfg.linear_conv_kernel_dim - 1)
    local_linear_k_heads = arch_cfg.linear_num_key_heads // tp
    local_linear_v_heads = arch_cfg.linear_num_value_heads // tp
    k_dim = local_linear_k_heads * arch_cfg.linear_key_head_dim
    v_dim = local_linear_v_heads * arch_cfg.linear_value_head_dim
    full_kv_dim = arch_cfg.num_key_value_heads * arch_cfg.head_dim
    ple_zero_based = {lid - 1 for lid in arch_cfg.ple_layer_ids}

    layer_plans: List[Qwen4_5LayerCachePlan] = []
    for layer_idx in range(arch_cfg.num_hidden_layers):
        ltype = arch_cfg.layer_types[layer_idx]
        is_consumer = layer_idx in share_map
        source_layer = share_map.get(layer_idx)
        opaque: List[OpaqueStateRegionPlan] = []

        allocates_kv = False
        kv_src: Optional[int] = None
        num_kv_heads = 0
        head_dim = 0
        kv_bpt = 0

        gdn_conv_c = 0
        gdn_conv_len = 0
        gdn_conv_bytes = 0
        allocates_ssm = False
        ssm_src: Optional[int] = None
        ssm_shape: Optional[Tuple[int, int, int]] = None
        ssm_bytes = 0

        has_kv_sconv = False
        k_sconv_shape: Optional[Tuple[int, int]] = None
        v_sconv_shape: Optional[Tuple[int, int]] = None
        has_qsa_indexer_cache = False
        indexer_kv_source_layer: Optional[int] = None
        indexer_head_dim = 0
        indexer_compress_ratio = 0

        if ltype == "full_attention":
            if not is_consumer:
                allocates_kv = True
                kv_src = layer_idx
                num_kv_heads = arch_cfg.num_key_value_heads
                head_dim = arch_cfg.head_dim
                kv_bpt = 2 * num_kv_heads * head_dim * kv_elem_bytes
                if arch_cfg.use_full_attn_kv_sconv and k_conv_state > 0:
                    has_kv_sconv = True
                    k_sconv_shape = (full_kv_dim, k_conv_state)
                    v_sconv_shape = (full_kv_dim, k_conv_state)
                    kv_sconv_shape = (2, full_kv_dim, k_conv_state)
                    opaque.append(
                        OpaqueStateRegionPlan(
                            tag=FULL_ATTN_KV_SCONV_TAG,
                            dtype=st_dt,
                            shape=kv_sconv_shape,
                            bytes_per_req=math.prod(kv_sconv_shape) * st_elem_bytes,
                            is_short_conv=True,
                            conv_tensor_count=2,
                        )
                    )
            else:
                allocates_kv = False
                kv_src = source_layer

            if arch_cfg.has_qsa:
                has_qsa_indexer_cache = not is_consumer
                indexer_kv_source_layer = layer_idx if not is_consumer else source_layer
                indexer_head_dim = int(arch_cfg.indexer_head_dim)
                indexer_compress_ratio = int(arch_cfg.indexer_compress_ratio)

            if (
                arch_cfg.has_output_sconv(
                    "full_attn_output_sconv", is_consumer=is_consumer
                )
                and k_conv_state > 0
            ):
                out_shape = (h, k_conv_state)
                opaque.append(
                    OpaqueStateRegionPlan(
                        tag=FULL_ATTN_OUTPUT_SCONV_TAG,
                        dtype=st_dt,
                        shape=out_shape,
                        bytes_per_req=math.prod(out_shape) * st_elem_bytes,
                        is_short_conv=True,
                        conv_tensor_count=1,
                    )
                )
        else:
            # Gated DeltaNet linear attention
            if not is_consumer:
                gdn_conv_c = 2 * k_dim + v_dim
                gdn_conv_len = k_conv_state
                gdn_conv_bytes = gdn_conv_c * gdn_conv_len * st_elem_bytes
                allocates_ssm = True
                ssm_src = layer_idx
            else:
                if not arch_cfg.gdn_no_q_conv and k_conv_state > 0:
                    gdn_conv_c = k_dim
                    gdn_conv_len = k_conv_state
                    gdn_conv_bytes = gdn_conv_c * gdn_conv_len * st_elem_bytes
                if use_yoco:
                    allocates_ssm = False
                    ssm_src = source_layer
                else:
                    allocates_ssm = True
                    ssm_src = layer_idx

            if allocates_ssm:
                ssm_shape = (
                    local_linear_v_heads,
                    arch_cfg.linear_key_head_dim,
                    arch_cfg.linear_value_head_dim,
                )
                ssm_bytes = math.prod(ssm_shape) * 4  # FP32 recurrent state

            if (
                arch_cfg.has_output_sconv(
                    "linear_attn_output_sconv", is_consumer=is_consumer
                )
                and k_conv_state > 0
            ):
                out_shape = (h, k_conv_state)
                opaque.append(
                    OpaqueStateRegionPlan(
                        tag=LINEAR_ATTN_OUTPUT_SCONV_TAG,
                        dtype=st_dt,
                        shape=out_shape,
                        bytes_per_req=math.prod(out_shape) * st_elem_bytes,
                        is_short_conv=True,
                        conv_tensor_count=1,
                    )
                )

        # PLE state (Layer 1, non-dilated conv_state [HC*H, K_ple - 1] + ngram_ctx [ngram_size - 1])
        has_ple = layer_idx in ple_zero_based
        ple_conv_shape: Optional[Tuple[int, int]] = None
        ple_ctx_shape: Optional[Tuple[int]] = None
        if has_ple:
            ple_conv_len = max(0, arch_cfg.ple_conv_kernel_size - 1)
            if ple_conv_len > 0:
                ple_conv_shape = (h_hc, ple_conv_len)
                opaque.append(
                    OpaqueStateRegionPlan(
                        tag=PLE_STATE_TAG,
                        dtype=st_dt,
                        shape=ple_conv_shape,
                        bytes_per_req=math.prod(ple_conv_shape) * st_elem_bytes,
                        is_short_conv=True,
                        conv_tensor_count=1,
                    )
                )
            ctx_len = arch_cfg.ngram_size - 1
            if ctx_len > 0:
                ple_ctx_shape = (ctx_len,)
                opaque.append(
                    OpaqueStateRegionPlan(
                        tag=PLE_NGRAM_CTX_TAG,
                        dtype="I64",
                        shape=ple_ctx_shape,
                        bytes_per_req=ctx_len * 8,
                        is_short_conv=False,
                        conv_tensor_count=0,
                    )
                )

        # FFN output short-conv (all 40 layers)
        if arch_cfg.use_ffn_output_sconv and k_conv_state > 0:
            ffn_shape = (h, k_conv_state)
            opaque.append(
                OpaqueStateRegionPlan(
                    tag=FFN_OUTPUT_SCONV_TAG,
                    dtype=st_dt,
                    shape=ffn_shape,
                    bytes_per_req=math.prod(ffn_shape) * st_elem_bytes,
                    is_short_conv=True,
                    conv_tensor_count=1,
                )
            )

        layer_plans.append(
            Qwen4_5LayerCachePlan(
                layer_idx=layer_idx,
                layer_type=ltype,
                is_sharing_consumer=is_consumer,
                sharing_source_layer=source_layer,
                allocates_kv_cache=allocates_kv,
                kv_source_layer=kv_src,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_bytes_per_token=kv_bpt,
                gdn_conv_channels=gdn_conv_c,
                gdn_conv_state_len=gdn_conv_len,
                gdn_conv_bytes_per_req=gdn_conv_bytes,
                allocates_ssm_state=allocates_ssm,
                ssm_source_layer=ssm_src,
                ssm_state_shape=ssm_shape,
                ssm_bytes_per_req=ssm_bytes,
                has_kv_sconv=has_kv_sconv,
                k_sconv_shape=k_sconv_shape,
                v_sconv_shape=v_sconv_shape,
                has_ple=has_ple,
                ple_conv_state_shape=ple_conv_shape,
                ple_ngram_ctx_shape=ple_ctx_shape,
                opaque_regions=tuple(opaque),
                has_qsa_indexer_cache=has_qsa_indexer_cache,
                indexer_kv_source_layer=indexer_kv_source_layer,
                indexer_head_dim=indexer_head_dim,
                indexer_compress_ratio=indexer_compress_ratio,
            )
        )

    return Qwen4_5KVCachePlan(
        arch_config=arch_cfg,
        enable_yoco_decode=use_yoco,
        kv_dtype=kv_dt,
        kv_dtype_bytes=kv_elem_bytes,
        layers=tuple(layer_plans),
        attention_sharing=share_map,
    )


@dataclass(frozen=True)
class DecodeOpaqueStatePoolPolicy:
    """Resolved pool/tail/reuse policy for the 9 Qwen 4.5 ``OPAQUE_STATE`` tags."""

    is_decode_role: bool
    rolling_enabled: bool
    mode: str
    group_type: Any
    enable_prefix_reuse: bool
    active_tail_blocks: int
    validate_tail_blocks: bool
    reservable: bool
    explicit_block_num: int
    charge_to_paged_budget: bool


def _parse_bool_env(raw: Optional[str], *, default: bool = True) -> bool:
    if raw is None or not raw.strip():
        return default
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag {raw!r} for {DECODE_STATE_ROLLING_ENV}")


def _parse_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}={raw!r}") from exc


def _resolve_is_decode_role(config: Any = None, role_type: Optional[Any] = None) -> bool:
    candidate = role_type
    if candidate is None and config is not None:
        par_cfg = getattr(config, "parallelism_config", None)
        if par_cfg is not None:
            candidate = getattr(par_cfg, "role_type", None)
        if candidate is None:
            candidate = getattr(config, "role_type", None)
    if candidate is None:
        env_role = os.environ.get("ROLE_TYPE", "").strip()
        candidate = env_role if env_role else None
    if candidate is None:
        return False
    if RoleType is not None and candidate == RoleType.DECODE:
        return True
    norm = str(getattr(candidate, "name", candidate)).strip().upper()
    return norm == "DECODE" or norm.endswith(".DECODE")


def _resolve_gen_num_per_cycle(config: Any = None) -> int:
    if config is not None:
        val = int(getattr(config, "gen_num_per_cycle", 0) or 0)
        if val > 0:
            return val
        sp_cfg = getattr(config, "sp_config", None)
        if sp_cfg is not None:
            val = int(getattr(sp_cfg, "gen_num_per_cycle", 0) or 0)
            if val > 0:
                return val
    env_val = _parse_int_env("GEN_NUM_PER_CYCLE")
    return max(0, env_val or 0)


def resolve_decode_opaque_state_pool_policy(
    config: Any = None,
    *,
    role_type: Optional[Any] = None,
    kv_cache_config: Optional[Any] = None,
    per_block_opaque_bytes: int = 0,
) -> DecodeOpaqueStatePoolPolicy:
    """Resolve rolling pool policy for Decode ``OPAQUE_STATE`` tags while preserving Prefill ``FULL``."""
    is_decode = _resolve_is_decode_role(config, role_type)
    full_gt = CacheGroupType.FULL if _HAS_RTP_OPS else "FULL"
    if not is_decode:
        return DecodeOpaqueStatePoolPolicy(
            is_decode_role=False,
            rolling_enabled=False,
            mode="full",
            group_type=full_gt,
            enable_prefix_reuse=True,
            active_tail_blocks=0,
            validate_tail_blocks=False,
            reservable=True,
            explicit_block_num=0,
            charge_to_paged_budget=True,
        )

    rolling_flag = _parse_bool_env(os.environ.get(DECODE_STATE_ROLLING_ENV), default=True)
    raw_mode = os.environ.get(DECODE_STATE_GROUP_TYPE_ENV, "").strip().lower()
    mode = raw_mode if raw_mode else "linear"
    if mode not in ("linear", "swa", "full"):
        raise ValueError(
            f"Unsupported {DECODE_STATE_GROUP_TYPE_ENV}={raw_mode!r}; "
            "expected 'linear', 'swa', or 'full'"
        )
    if not rolling_flag or mode == "full":
        return DecodeOpaqueStatePoolPolicy(
            is_decode_role=True,
            rolling_enabled=False,
            mode="full",
            group_type=full_gt,
            enable_prefix_reuse=True,
            active_tail_blocks=0,
            validate_tail_blocks=False,
            reservable=True,
            explicit_block_num=0,
            charge_to_paged_budget=True,
        )

    env_pool = _parse_int_env(DECODE_STATE_POOL_BLOCKS_ENV)
    if env_pool is not None and (env_pool < 0 or env_pool == 1):
        raise ValueError(
            f"{DECODE_STATE_POOL_BLOCKS_ENV} must be >= 2 (or 0 for auto), got {env_pool}"
        )

    cfg_pool = 0
    if kv_cache_config is not None:
        pool_attr = "linear_pool_blocks" if mode == "linear" else "swa_pool_blocks"
        cfg_pool = int(getattr(kv_cache_config, pool_attr, 0) or 0)
        if cfg_pool < 0 or cfg_pool == 1:
            raise ValueError(
                f"kv_cache_config.{pool_attr} must be >= 2 (or 0 for auto), got {cfg_pool}"
            )

    if env_pool is not None and env_pool >= 2:
        pool_blocks = env_pool
    elif cfg_pool >= 2:
        pool_blocks = cfg_pool
    else:
        gen_num_per_cycle = _resolve_gen_num_per_cycle(config)
        pool_blocks = (
            DEFAULT_DECODE_SPEC_LINEAR_POOL_BLOCKS
            if (mode == "linear" and gen_num_per_cycle > 0)
            else DEFAULT_DECODE_OPAQUE_POOL_BLOCKS
        )
        if kv_cache_config is not None:
            test_block_num = int(getattr(kv_cache_config, "test_block_num", 0) or 0)
            if test_block_num > 0:
                pool_blocks = max(2, min(pool_blocks, test_block_num))
            mem_mb = int(getattr(kv_cache_config, "kv_cache_mem_mb", 0) or 0)
            if mem_mb > 0 and per_block_opaque_bytes > 0:
                budget_bytes = mem_mb * 1024 * 1024
                max_auto = max(2, (budget_bytes // 2) // int(per_block_opaque_bytes))
                pool_blocks = max(2, min(pool_blocks, max_auto))

    tail_override = _parse_int_env(DECODE_STATE_ACTIVE_TAIL_BLOCKS_ENV)
    if tail_override is not None and tail_override < 1:
        raise ValueError(
            f"{DECODE_STATE_ACTIVE_TAIL_BLOCKS_ENV} must be >= 1 in rolling mode, got {tail_override}"
        )
    # LINEAR retains max(2, activeTailBlocks()) + reserve_step in LinearCacheManager while loading 1 tail block;
    # SWA evicts blocks outside activeTailBlocks() + reserve_step, so default to 2 blocks across TPB boundaries.
    active_tail = (
        tail_override
        if tail_override is not None
        else (1 if mode == "linear" else 2)
    )
    gt = (
        (CacheGroupType.LINEAR if mode == "linear" else CacheGroupType.SWA)
        if _HAS_RTP_OPS
        else mode.upper()
    )
    return DecodeOpaqueStatePoolPolicy(
        is_decode_role=True,
        rolling_enabled=True,
        mode=mode,
        group_type=gt,
        enable_prefix_reuse=False,
        active_tail_blocks=active_tail,
        validate_tail_blocks=False,
        reservable=False,
        explicit_block_num=int(pool_blocks),
        charge_to_paged_budget=True,
    )


def _make_opaque_state_desc(
    tag: str,
    entry_elems: int,
    entry_count: int,
    dtype_str: str,
    *,
    pool_policy: Optional[DecodeOpaqueStatePoolPolicy] = None,
) -> Any:
    """Create an EXPLICIT ``OPAQUE_STATE`` ``KVCacheSpecDesc`` (FULL on Prefill, rolling on Decode)."""
    dtype_map = {
        "BF16": DataType.TYPE_BF16,
        "F16": DataType.TYPE_FP16,
        "FP16": DataType.TYPE_FP16,
        "F32": DataType.TYPE_FP32,
        "I64": DataType.TYPE_INT64,
    }
    dt = dtype_map[dtype_str]
    desc = KVCacheSpecDesc()
    desc.tag = tag
    desc.cache_type = KVCacheSpecType.OPAQUE_STATE
    desc.is_state_cache = True
    desc.dtype = dt
    desc.entry_dtype = dt
    desc.entry_elems = entry_elems
    desc.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
    desc.explicit_entry_count = entry_count
    if pool_policy is not None and pool_policy.rolling_enabled:
        desc.group_type = pool_policy.group_type
        reuse = CacheReusePolicyDesc()
        reuse.enable_prefix_reuse = bool(pool_policy.enable_prefix_reuse)
        desc.reuse = reuse
        tail = CacheTailPolicyDesc()
        tail.active_tail_blocks = int(pool_policy.active_tail_blocks)
        tail.validate_tail_blocks = bool(pool_policy.validate_tail_blocks)
        desc.tail = tail
        cap = CacheCapacityPolicyDesc()
        cap.reservable = bool(pool_policy.reservable)
        cap.explicit_block_num = int(pool_policy.explicit_block_num)
        cap.charge_to_paged_budget = bool(pool_policy.charge_to_paged_budget)
        desc.capacity = cap
    else:
        desc.group_type = CacheGroupType.FULL
        reuse = CacheReusePolicyDesc()
        reuse.enable_prefix_reuse = True
        desc.reuse = reuse
        tail = CacheTailPolicyDesc()
        tail.active_tail_blocks = 0
        desc.tail = tail
    desc.tp_transfer_mode = TpCacheTransferMode.REPLICATED
    return desc


def _make_full_group_opaque_desc(
    tag: str,
    entry_elems: int,
    entry_count: int,
    dtype_str: str,
) -> Any:
    """Create a FULL-group EXPLICIT ``OPAQUE_STATE`` ``KVCacheSpecDesc``."""
    return _make_opaque_state_desc(tag, entry_elems, entry_count, dtype_str)


def build_qwen4_5_kv_cache_spec_descs(
    config: Any,
    arch_cfg: Optional[Qwen4_5ArchConfig] = None,
    *,
    tp_size: int = 1,
    parallel_mode: Optional[str] = None,
    kv_dtype: str = "BF16",
    enable_yoco_decode: Optional[bool] = None,
    allow_small_block_explicit: bool = False,
    role_type: Optional[Any] = None,
    kv_cache_config: Optional[Any] = None,
) -> List[List[Any]]:
    """Build per-layer RTP-LLM ``KVCacheSpecDesc`` lists mirroring ``qwen4_exp_kv_cache``.

    Requires compiled ``rtp_llm.ops``; for pure-Python environments use
    :func:`build_qwen4_5_kv_cache_plan`.
    """
    if not _HAS_RTP_OPS:
        raise ImportError(
            "build_qwen4_5_kv_cache_spec_descs requires compiled rtp_llm.ops; "
            "use build_qwen4_5_kv_cache_plan() in pure-Python mode."
        )

    resolved_arch = arch_cfg
    if resolved_arch is None:
        resolved_arch = getattr(config, "qwen4_5_arch_config", None) or getattr(
            config, "arch_config", None
        )
    if resolved_arch is None:
        raise ValueError("build_qwen4_5_kv_cache_spec_descs requires Qwen4_5ArchConfig")

    plan = build_qwen4_5_kv_cache_plan(
        resolved_arch,
        tp_size=tp_size,
        parallel_mode=parallel_mode,
        kv_dtype=kv_dtype,
        enable_yoco_decode=enable_yoco_decode,
    )
    pool_policy = resolve_decode_opaque_state_pool_policy(
        config,
        role_type=role_type,
        kv_cache_config=kv_cache_config,
        per_block_opaque_bytes=plan.opaque_state_bytes_per_block,
    )

    layer_descs: List[List[Any]] = []
    for lp in plan.layers:
        descs: List[Any] = []
        if lp.layer_type == "full_attention":
            if lp.allocates_kv_cache:
                mha = KVCacheSpecDesc()
                mha.tag = FULL_ATTN_KV_TAG
                mha.cache_type = KVCacheSpecType.MHA
                mha.tp_transfer_mode = TpCacheTransferMode.MHA_HEADS
                descs.append(mha)
            if lp.has_qsa_indexer_cache:
                attn_cfg = getattr(config, "attn_config", None)
                kernel_tpb = (
                    int(getattr(attn_cfg, "kernel_tokens_per_block", 0) or 0)
                    if attn_cfg is not None
                    else 0
                )
                phys_tpb = (
                    int(getattr(attn_cfg, "tokens_per_block", 0) or 0)
                    if attn_cfg is not None
                    else 0
                )
                effective_tpb = kernel_tpb if kernel_tpb > 0 else phys_tpb
                if allow_small_block_explicit and 0 < effective_tpb < 128:
                    if phys_tpb > 0 and kernel_tpb > 0 and phys_tpb != kernel_tpb:
                        raise ValueError(
                            f"QSA small-block explicit mode requires tokens_per_block ({phys_tpb}) "
                            f"== kernel_tokens_per_block ({kernel_tpb})"
                        )
                    if effective_tpb % lp.indexer_compress_ratio != 0:
                        raise ValueError(
                            f"tokens_per_block ({effective_tpb}) must be divisible by "
                            f"indexer_compress_ratio ({lp.indexer_compress_ratio})"
                        )
                    idx_kv = KVCacheSpecDesc()
                    idx_kv.tag = INDEXER_KV_TAG
                    idx_kv.cache_type = KVCacheSpecType.OPAQUE_KV
                    idx_kv.is_state_cache = False
                    idx_kv.dtype = DataType.TYPE_UINT8
                    idx_kv.entry_dtype = DataType.TYPE_UINT8
                    idx_kv.entry_elems = lp.indexer_head_dim * 2
                    idx_kv.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
                    idx_kv.explicit_entry_count = (
                        effective_tpb // lp.indexer_compress_ratio
                    )
                    idx_kv.compression_ratio = lp.indexer_compress_ratio
                    idx_reuse = CacheReusePolicyDesc()
                    idx_reuse.enable_prefix_reuse = True
                    idx_kv.reuse = idx_reuse
                    idx_tail = CacheTailPolicyDesc()
                    idx_tail.active_tail_blocks = 0
                    idx_kv.tail = idx_tail
                else:
                    idx_kv = indexer_kv_desc(
                        lp.indexer_head_dim, lp.indexer_compress_ratio
                    )
                idx_kv.tp_transfer_mode = TpCacheTransferMode.REPLICATED
                descs.append(idx_kv)

                idx_state = indexer_state_desc(
                    lp.indexer_head_dim, lp.indexer_compress_ratio
                )
                idx_state.tp_transfer_mode = TpCacheTransferMode.REPLICATED
                descs.append(idx_state)
        else:
            if lp.allocates_ssm_state:
                assert lp.ssm_state_shape is not None
                ssm_heads, ssm_k, ssm_v = lp.ssm_state_shape
                descs.append(
                    _make_opaque_state_desc(
                        tag=GDN_SSM_STATE_TAG,
                        entry_elems=ssm_k * ssm_v,
                        entry_count=ssm_heads,
                        dtype_str="F32",
                        pool_policy=pool_policy,
                    )
                )
                if lp.has_gdn_conv:
                    descs.append(
                        _make_opaque_state_desc(
                            tag=GDN_CONV_STATE_TAG,
                            entry_elems=lp.gdn_conv_channels,
                            entry_count=lp.gdn_conv_state_len,
                            dtype_str="BF16",
                            pool_policy=pool_policy,
                        )
                    )
            elif lp.has_gdn_conv:
                # YoCo decode consumer: no SSM state, only Q-conv [2048, k-1=3]
                descs.append(
                    _make_opaque_state_desc(
                        tag=GDN_CONSUMER_Q_CONV_TAG,
                        entry_elems=lp.gdn_conv_channels,
                        entry_count=lp.gdn_conv_state_len,
                        dtype_str="BF16",
                        pool_policy=pool_policy,
                    )
                )

        for reg in lp.opaque_regions:
            if reg.tag == PLE_NGRAM_CTX_TAG:
                entry_elems = 1
                entry_count = reg.shape[0]
            elif reg.tag == FULL_ATTN_KV_SCONV_TAG:
                # Packed (2, kv_dim, k-1) -> 2*kv_dim elems per history step
                entry_elems = reg.shape[0] * reg.shape[1]
                entry_count = reg.shape[2]
            else:
                # (channels, k-1) -> channels elems per history step
                entry_elems = reg.shape[0]
                entry_count = reg.shape[1]
            descs.append(
                _make_opaque_state_desc(
                    tag=reg.tag,
                    entry_elems=entry_elems,
                    entry_count=entry_count,
                    dtype_str=reg.dtype,
                    pool_policy=pool_policy,
                )
            )
        layer_descs.append(descs)

    return layer_descs


@dataclass(frozen=True)
class DFlash2KVCachePlan:
    """Per-token and per-request KV cache sizing descriptor for DFlash2."""

    num_hidden_layers: int
    num_kv_heads: int
    head_dim: int
    kv_dtype: str
    kv_dtype_bytes: int
    sliding_window: Optional[int]
    kv_bytes_per_token: int

    def total_bytes_for_seq_len(self, seq_len: int, batch_size: int = 1) -> int:
        if seq_len < 0 or batch_size < 0:
            raise ValueError("seq_len and batch_size must be non-negative")
        return int(batch_size) * int(seq_len) * int(self.kv_bytes_per_token)


def dflash2_kv_dtype_to_str(dflash_kv_dtype: str = "bf16") -> str:
    norm = str(dflash_kv_dtype or "bf16").strip().upper()
    if norm in ("FP8_E4M3", "F8_E4M3", "FP8"):
        return "F8_E4M3"
    if norm in ("BF16", "BFLOAT16"):
        return "BF16"
    if norm in ("FP16", "F16", "FLOAT16"):
        return "F16"
    raise ValueError(f"Unsupported dflash_kv_dtype={dflash_kv_dtype!r}")


def build_dflash2_kv_cache_plan(
    spec: Any,
    *,
    parallel_mode: Optional[str] = None,
    tp_size: int = 1,
    dflash_kv_dtype: str = "bf16",
) -> DFlash2KVCachePlan:
    del parallel_mode, tp_size  # DFlash2 is MQA-1 (1 KV head, replicated on every TP rank).
    dt_str = dflash2_kv_dtype_to_str(dflash_kv_dtype)
    elem_bytes = _KV_DTYPE_BYTES[dt_str]
    n_layers = int(getattr(spec, "num_hidden_layers", 5))
    n_kv = int(getattr(spec, "num_key_value_heads", 1))
    hd = int(getattr(spec, "head_dim", 256))
    swa = getattr(spec, "swa_window_size", None)
    kv_bpt = n_layers * 2 * n_kv * hd * elem_bytes
    return DFlash2KVCachePlan(
        num_hidden_layers=n_layers,
        num_kv_heads=n_kv,
        head_dim=hd,
        kv_dtype=dt_str,
        kv_dtype_bytes=elem_bytes,
        sliding_window=int(swa) if swa is not None else None,
        kv_bytes_per_token=kv_bpt,
    )
