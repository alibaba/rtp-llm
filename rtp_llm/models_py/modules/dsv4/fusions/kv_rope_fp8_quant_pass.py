"""GraphFX scaffold for DSV4 Path2 KV-compress/RoPE -> FP8 quant fusion.

Scenario
========
After the Path2 compressor writer has already fused KV compress, RMSNorm,
RoPE, and cache/indexer insertion, a remaining standalone FP8 quant can still
show up before a DeepGEMM consumer:

    _fused_kv_compress_norm_rope_insert_indexer_attn
            |
            v
    bf16 kv/indexer tensor
            |
            v
    sgl_per_token_group_quant_fp8
            |
            v
    (fp8, scale) -> DeepGEMM

The intended final replacement is a producer-side dual-output kernel:

    _fused_kv_compress_norm_rope_insert_indexer_attn_with_fp8_quant
            |
            +--> bf16 kv/indexer tensor -> existing cache/indexer consumers
            |
            +--> fp8, scale -----------> DeepGEMM

This pass deliberately does not invent the dual-output CUDA payload.  It only
installs the GraphFX/provenance shape needed by that payload and is gated by
``DSV4_KV_ROPE_FP8_QUANT``.  Without valid producer provenance the runtime
wrapper falls back to the original quant or raises when
``DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE=1`` is set, so the pass cannot silently
claim a launch reduction it did not obtain.

Code key paths
==============
GraphFX install/registry:
``rtp_llm/models_py/modules/dsv4/fusions/graphfx_injector.py`` and
``rtp_llm/models_py/modules/dsv4/fusions/fusion_registry.py``.

Path2 producer:
``rtp_llm/models_py/modules/dsv4/fp8/_compressor_vllm_triton.py``.

Potential FP8 consumers:
``rtp_llm/models_py/modules/dsv4/qlinear.py`` and
``rtp_llm/models_py/modules/dsv4/decode/indexer_decode_op.py``.
"""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import (
    eliminate_dead_code_preserving_dsv4_side_effects,
    record_dsv4_fusion_hit,
    record_dsv4_fusion_miss,
    register_dsv4_fusion_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_runtime import (
    dsv4_kv_rope_fp8_quant_from_provenance,
    dsv4_kv_rope_quant_producer_token,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_fp8_quant_pass import (
    _is_quant_node,
    _quant_contract_ok,
    _target_name,
    _unwrap_layout_only,
)

logger = logging.getLogger(__name__)

_PRODUCER_NAME_TOKENS = (
    "fused_kv_compress_norm_rope_insert_indexer_attn",
    "fused_kv_compress_norm_rope_insert",
    "run_fused_compress_kv_write_bf16",
    "kv_rope_quant_producer",
)
_VALUE_LABEL_TOKENS = (
    "kv_compress",
    "kv_indexer",
    "indexer_attn",
    "compress_norm_rope",
)


def _node_label(node: object, seen: set[torch.fx.Node] | None = None) -> str:
    if not isinstance(node, torch.fx.Node):
        return str(node).lower()
    if seen is None:
        seen = set()
    if node in seen:
        return ""
    seen.add(node)
    arg_labels = " ".join(_node_label(arg, seen) for arg in node.args)
    return " ".join(
        str(item).lower()
        for item in (
            getattr(node, "name", ""),
            getattr(node, "target", ""),
            node,
            arg_labels,
        )
    )


def _is_layout_only_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_method" and _target_name(node.target) in (
        "contiguous",
        "reshape",
        "view",
    ):
        return True
    return node.op == "call_function" and _target_name(node.target) in (
        "reshape",
        "view",
    )


def _all_users_are_layout_outputs(
    node: torch.fx.Node,
    seen: set[torch.fx.Node] | None = None,
) -> bool:
    if seen is None:
        seen = set()
    if node in seen:
        return True
    seen.add(node)
    if not node.users:
        return False
    for user in list(node.users):
        if user.op == "output":
            continue
        if not _is_layout_only_node(user):
            return False
        if not _all_users_are_layout_outputs(user, seen):
            return False
    return True


def _is_kv_rope_producer_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    target = _target_name(node.target).lower()
    return any(token in target for token in _PRODUCER_NAME_TOKENS)


def _looks_like_kv_rope_value(node: object) -> bool:
    label = _node_label(node)
    return any(token in label for token in _VALUE_LABEL_TOKENS)


def _rewrite_quant_from_kv_rope_provenance(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> bool:
    if not _is_quant_node(node):
        return False
    if not _quant_contract_ok(node):
        record_dsv4_fusion_miss("kv_rope_fp8_quant_fx", "kv_rope_quant_contract_mismatch")
        return False
    quant_arg = node.args[0] if node.args else None
    x_node = _unwrap_layout_only(quant_arg)
    if not isinstance(x_node, torch.fx.Node):
        record_dsv4_fusion_miss("kv_rope_fp8_quant_fx", "kv_rope_quant_unknown_producer")
        return False
    if not (_is_kv_rope_producer_node(x_node) or _looks_like_kv_rope_value(x_node)):
        record_dsv4_fusion_miss("kv_rope_fp8_quant_fx", "kv_rope_quant_unknown_producer")
        return False
    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(dsv4_kv_rope_quant_producer_token, args=(x_node,))
        fused = gm.graph.call_function(
            dsv4_kv_rope_fp8_quant_from_provenance,
            args=(token,),
            kwargs={
                "fallback_y": quant_arg if quant_arg is not x_node else None,
                "group_size": node.kwargs.get("group_size", 128),
                "eps": node.kwargs.get("eps", 1e-4),
                "column_major_scales": node.kwargs.get("column_major_scales", False),
                "scale_tma_aligned": node.kwargs.get("scale_tma_aligned", False),
                "scale_ue8m0": node.kwargs.get("scale_ue8m0", False),
                "fuse_silu_and_mul": node.kwargs.get("fuse_silu_and_mul", False),
                "masked_m": node.kwargs.get("masked_m", None),
            },
        )
    node.replace_all_uses_with(fused)
    return True


def _insert_kv_rope_producer_tokens(gm: torch.fx.GraphModule) -> int:
    replaced = 0
    for node in list(gm.graph.nodes):
        if not _is_kv_rope_producer_node(node):
            continue
        if any(
            isinstance(user, torch.fx.Node)
            and user.op == "call_function"
            and user.target is dsv4_kv_rope_quant_producer_token
            for user in node.users
        ):
            continue
        if not _all_users_are_layout_outputs(node):
            record_dsv4_fusion_miss(
                "kv_rope_fp8_quant_fx",
                "kv_rope_quant_has_bf16_consumer_without_dual_output",
            )
            continue
        with gm.graph.inserting_after(node):
            token = gm.graph.call_function(dsv4_kv_rope_quant_producer_token, args=(node,))
        for user in list(node.users):
            if user is token:
                continue
            user.replace_input_with(node, token)
        replaced += 1
    return replaced


def apply_kv_rope_fp8_quant_fx_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    consumers = 0
    for node in list(gm.graph.nodes):
        if _rewrite_quant_from_kv_rope_provenance(gm, node):
            consumers += 1
    producers = _insert_kv_rope_producer_tokens(gm)
    if consumers or producers:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
        record_dsv4_fusion_hit("kv_rope_fp8_quant_fx", consumers + producers)
        logger.info(
            "DSV4 FX KV RoPE+FP8 quant pass: consumers=%d producer_tokens=%d",
            consumers,
            producers,
        )
    return gm


def register_kv_rope_fp8_quant_pass() -> None:
    register_dsv4_fusion_pass(
        "kv_rope_fp8_quant_fx",
        apply_kv_rope_fp8_quant_fx_pass,
        priority=20,
        env_gate="DSV4_KV_ROPE_FP8_QUANT",
    )


register_kv_rope_fp8_quant_pass()
