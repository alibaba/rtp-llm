"""GraphFX replacement for DSV4 RMSNorm with both BF16 and FP8 consumers.

Scenario
========
Some DSV4 RMSNorm sites feed a BF16 path and an FP8 DeepGEMM path at the same
time.  The unfused graph launches RMSNorm first, then launches a standalone
per-token-group FP8 quant for the DeepGEMM input:

    x --------+
             v
    weight -> rmsnorm(x, weight, norm_eps) -> y_bf16 --+--> BF16 consumer
                                                       |
                                                       v
                              sgl_per_token_group_quant_fp8(y_bf16)
                                                       |
                                                       v
                                             (x_fp8, x_scale) -> DeepGEMM

Because the BF16 value is still needed, the plain ``rmsnorm_fp8_quant_fx`` pass
must not replace it with an FP8-only result.  This pass uses the dual-output
CUDA wrapper instead:

    x --------+
             +--> fused_rmsnorm_bf16_fp8_quant(x, weight)
    weight --+                         |
                                      +--> y_bf16 -> BF16 consumer
                                      |
                                      +--> x_fp8, x_scale -> DeepGEMM

The same rule also covers mutating RMSNorm call sites where the pybind op
writes into a preallocated output tensor.  FX matching treats that output
tensor as the BF16 value and replaces all non-quant users with the fused
BF16 output while replacing quant tuple users with the fused FP8/scale outputs.

Code key paths
==============
GraphFX install/registry:
``rtp_llm/models_py/modules/dsv4/fusions/graphfx_injector.py`` and
``rtp_llm/models_py/modules/dsv4/fusions/fusion_registry.py``.

Shared RMSNorm/quant match helpers:
``rtp_llm/models_py/modules/dsv4/fusions/rmsnorm_fp8_quant_pass.py``.

CUDA wrapper:
``rtp_llm/models_py/kernels/cuda/fused_rmsnorm_fp8_quant``.

Concrete DSV4 origins as of 2026-05-18
======================================
The hot producer is the FP8 attention Q-LoRA norm:
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:1739-1748`` builds the
mutating RMSNorm form
``out = torch.empty_like(x_2d); rtp_llm_ops.rmsnorm(out, x_2d, weight, eps,
stream); return out.view(orig_shape)``.

Decode path:
``rtp_llm/models_py/modules/dsv4/fp8/decode/compute_qkv.py:57-74`` computes
``qr = attn._rmsnorm_weighted(attn._lin(attn.wq_a, x), attn.q_norm)``, uses
``qr`` as the input to ``attn.wq_b``, and returns ``DecodeQKV.qr`` as the BF16
value.

Prefill path:
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:3884-3920`` computes the
same ``qr``, consumes it in ``self.wq_b``, and returns ``PrefillQKV.qr``.

BF16 consumers:
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:1991`` passes decode
``qkv.qr`` to ``IndexerFP8.forward_decode_vectorized`` and
``attention.py:2300`` passes prefill ``qkv.qr`` to ``IndexerFP8``.

FP8 quant consumers:
``rtp_llm/models_py/modules/factory/linear/impl/cuda/fp8_deepgemm_linear.py:215``
is the DeepGEMM quant contract for ``attn.wq_b`` and, when the indexer is
captured in the same graph, ``rtp_llm/models_py/modules/dsv4/fp8/indexer.py:359``
and ``indexer.py:378-386`` provide a second ``wq_b`` quant consumer.  This pass
rewrites one visible quant node at a time; if a second FP8 consumer remains in
the same FX graph it stays correct by quantizing the fused BF16 output.
"""

from __future__ import annotations

import logging
import operator

import torch

from rtp_llm.models_py.kernels.cuda.fused_rmsnorm_fp8_quant import (
    fused_rmsnorm_bf16_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import (
    eliminate_dead_code_preserving_dsv4_side_effects,
    record_dsv4_fusion_hit,
    record_dsv4_fusion_miss,
    register_dsv4_fusion_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_fp8_quant_pass import (
    _all_users_are_layout_outputs,
    _graph_output_users,
    _hidden_contract_ok,
    _is_layout_only_node,
    _is_plain_rmsnorm_node,
    _is_quant_node,
    _node_label,
    _quant_contract_ok,
    _rmsnorm_args,
    _target_name,
    _unwrap_layout_only,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_quant_runtime import (
    dsv4_rmsnorm_bf16_fp8_quant_from_provenance,
    dsv4_rmsnorm_bf16_fp8_quant_mutating_producer_token,
    dsv4_rmsnorm_bf16_fp8_quant_producer_token,
)

logger = logging.getLogger(__name__)

_PRODUCER_SCOPE_DENY_SUBSTRINGS = (
    "final_ln",
    "final_norm",
    "input_layernorm",
    "post_attention_layernorm",
)


def _is_getitem(node: object, source: torch.fx.Node, index: int) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and node.args[0] is source
        and node.args[1] == index
    )


def _replace_quant_uses(quant_node: torch.fx.Node, q_node: torch.fx.Node, s_node: torch.fx.Node) -> None:
    for user in list(quant_node.users):
        if _is_getitem(user, quant_node, 0):
            user.replace_all_uses_with(q_node)
        elif _is_getitem(user, quant_node, 1):
            user.replace_all_uses_with(s_node)
        else:
            raise RuntimeError(
                "DSV4 RMSNorm BF16+FP8 quant pass expects quant tuple users "
                f"to be operator.getitem(0/1), got target={_target_name(user.target)}"
            )


def _is_quant_only_layout_user(user: torch.fx.Node, quant_node: torch.fx.Node) -> bool:
    if user is quant_node:
        return True
    if user.op == "call_method" and _target_name(user.target) in ("contiguous", "reshape", "view"):
        return bool(user.users) and all(_is_quant_only_layout_user(next_user, quant_node) for next_user in user.users)
    if user.op == "call_function" and _target_name(user.target) in ("reshape", "view"):
        return bool(user.users) and all(_is_quant_only_layout_user(next_user, quant_node) for next_user in user.users)
    return False


def _is_mutating_rmsnorm_node(node: object, output_node: torch.fx.Node) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and "rmsnorm" in _target_name(node.target).lower()
        and len(node.args) >= 4
        and node.args[0] is output_node
    )


def _producer_scope_ok(x: object, weight: object) -> bool:
    text = f"{_node_label(x)} {_node_label(weight)}"
    if any(token in text for token in _PRODUCER_SCOPE_DENY_SUBSTRINGS):
        record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "producer_scope_denied")
        return False
    padded = f" {text} "
    if (
        "l_weight_" not in text
        and "self_weight" not in text
        and " weight " not in padded
        and "_weight" not in text
    ):
        record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "producer_scope_not_attention_local")
        return False
    return True


def _all_users_are_layout_outputs_except(
    node: torch.fx.Node,
    skip_users: set[torch.fx.Node],
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
        if user in skip_users or user.op == "output":
            continue
        if not _is_layout_only_node(user):
            return False
        if not _all_users_are_layout_outputs_except(user, skip_users, seen):
            return False
    return True


def _match_rmsnorm_value(node: object):
    if not isinstance(node, torch.fx.Node):
        return None
    if _is_plain_rmsnorm_node(node):
        x, weight, norm_eps = _rmsnorm_args(node)
        return {
            "value": node,
            "insert_before": node,
            "skip_users": set(),
            "x": x,
            "weight": weight,
            "norm_eps": norm_eps,
        }
    for user in list(node.users):
        if _is_mutating_rmsnorm_node(user, node):
            return {
                "value": node,
                "insert_before": node,
                "skip_users": {user},
                "x": user.args[1],
                "weight": user.args[2],
                "norm_eps": user.args[3],
            }
    return None


def _has_bf16_consumer(value_node: torch.fx.Node, quant_node: torch.fx.Node, skip_users: set[torch.fx.Node]) -> bool:
    return any(
        user not in skip_users and not _is_quant_only_layout_user(user, quant_node)
        for user in value_node.users
    )


def _replace_bf16_uses(
    value_node: torch.fx.Node,
    replacement: torch.fx.Node,
    quant_node: torch.fx.Node,
    skip_users: set[torch.fx.Node],
) -> None:
    for user in list(value_node.users):
        if user in skip_users or _is_quant_only_layout_user(user, quant_node):
            continue
        user.replace_input_with(value_node, replacement)


def _erase_dead_nodes(nodes: set[torch.fx.Node]) -> None:
    for node in list(nodes):
        if len(node.users) == 0:
            node.graph.erase_node(node)


def _insert_functional_rmsnorm_token(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_plain_rmsnorm_node(node) or len(node.users) == 0:
        return False
    x, weight, norm_eps = _rmsnorm_args(node)
    if not _hidden_contract_ok(x, weight):
        record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "producer_unsupported_fixed_hidden_dim")
        return False
    if not _producer_scope_ok(x, weight):
        return False
    if not _all_users_are_layout_outputs(node):
        return False
    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(
            dsv4_rmsnorm_bf16_fp8_quant_producer_token,
            args=(x, weight, norm_eps),
        )
    node.replace_all_uses_with(token)
    if len(node.users) == 0:
        gm.graph.erase_node(node)
    return True


def _insert_mutating_rmsnorm_token(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_plain_rmsnorm_node(node) or len(node.args) < 4:
        return False
    output_node, x, weight, norm_eps = node.args[:4]
    if not isinstance(output_node, torch.fx.Node):
        return False
    if not _hidden_contract_ok(x, weight):
        record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "producer_unsupported_fixed_hidden_dim")
        return False
    if not _producer_scope_ok(x, weight):
        return False
    if not _graph_output_users(output_node) and not any(
        user is not node and _is_layout_only_node(user) for user in output_node.users
    ):
        return False
    if not _all_users_are_layout_outputs_except(output_node, {node}):
        return False
    with gm.graph.inserting_before(node):
        gm.graph.call_function(
            dsv4_rmsnorm_bf16_fp8_quant_mutating_producer_token,
            args=(output_node, x, weight, norm_eps),
        )
    if len(node.users) == 0:
        gm.graph.erase_node(node)
    return True


def _insert_rmsnorm_producer_tokens(gm: torch.fx.GraphModule) -> int:
    replaced = 0
    for node in list(gm.graph.nodes):
        if not isinstance(node, torch.fx.Node):
            continue
        if _insert_mutating_rmsnorm_token(gm, node):
            replaced += 1
            continue
        if _insert_functional_rmsnorm_token(gm, node):
            replaced += 1
    return replaced


def _rewrite_quant_from_provenance(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_quant_node(node):
        return False
    if not _quant_contract_ok(node):
        record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "cross_graph_quant_contract_mismatch")
        return False
    x_node = _unwrap_layout_only(node.args[0] if node.args else None)
    if _match_rmsnorm_value(x_node) is not None:
        return False
    if not isinstance(x_node, torch.fx.Node) or x_node.op != "placeholder":
        record_dsv4_fusion_miss(
            "rmsnorm_bf16_fp8_quant_fx", "cross_graph_quant_input_not_graph_input"
        )
        return False
    quant_eps = node.kwargs.get("eps", 1e-10)
    provenance_input = x_node
    fallback_input = node.args[0] if node.args else None
    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            dsv4_rmsnorm_bf16_fp8_quant_from_provenance,
            args=(provenance_input,),
            kwargs={
                "fallback_y": fallback_input if fallback_input is not provenance_input else None,
                "group_size": node.kwargs.get("group_size", 128),
                "eps": quant_eps,
                "column_major_scales": node.kwargs.get("column_major_scales", False),
                "scale_tma_aligned": node.kwargs.get("scale_tma_aligned", False),
                "scale_ue8m0": node.kwargs.get("scale_ue8m0", False),
                "fuse_silu_and_mul": node.kwargs.get("fuse_silu_and_mul", False),
                "masked_m": node.kwargs.get("masked_m", None),
            },
        )
    node.replace_all_uses_with(fused)
    return True


def apply_rmsnorm_bf16_fp8_quant_fx_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    same_graph = 0
    cross_graph_consumers = 0
    for node in list(gm.graph.nodes):
        if not _is_quant_node(node):
            continue
        if not _quant_contract_ok(node):
            record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "quant_contract_mismatch")
            continue
        x_node = _unwrap_layout_only(node.args[0] if node.args else None)
        rmsnorm = _match_rmsnorm_value(x_node)
        if rmsnorm is None:
            record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "quant_input_not_rmsnorm")
            continue
        value_node = rmsnorm["value"]
        skip_users = rmsnorm["skip_users"]
        if not _has_bf16_consumer(value_node, node, skip_users):
            record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "no_bf16_consumer")
            continue
        x, weight, norm_eps = rmsnorm["x"], rmsnorm["weight"], rmsnorm["norm_eps"]
        if not _hidden_contract_ok(x, weight):
            record_dsv4_fusion_miss("rmsnorm_bf16_fp8_quant_fx", "unsupported_fixed_hidden_dim")
            continue
        quant_eps = node.kwargs.get("eps", 1e-10)
        with gm.graph.inserting_before(rmsnorm["insert_before"]):
            fused = gm.graph.call_function(
                fused_rmsnorm_bf16_fp8_quant,
                args=(x, weight),
                kwargs={"norm_eps": norm_eps, "quant_eps": quant_eps, "group_size": 128},
            )
            y_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
            q_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
            s_node = gm.graph.call_function(operator.getitem, args=(fused, 2))
        _replace_quant_uses(node, q_node, s_node)
        _replace_bf16_uses(value_node, y_node, node, skip_users)
        _erase_dead_nodes(skip_users)
        same_graph += 1
    if same_graph:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
    for node in list(gm.graph.nodes):
        if _rewrite_quant_from_provenance(gm, node):
            cross_graph_consumers += 1
    cross_graph_producers = _insert_rmsnorm_producer_tokens(gm)
    if same_graph or cross_graph_consumers or cross_graph_producers:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
        record_dsv4_fusion_hit(
            "rmsnorm_bf16_fp8_quant_fx",
            same_graph + cross_graph_consumers + cross_graph_producers,
        )
        logger.info(
            "DSV4 FX RMSNorm BF16+FP8 quant pass: same_graph=%d "
            "cross_graph_producers=%d cross_graph_consumers=%d",
            same_graph,
            cross_graph_producers,
            cross_graph_consumers,
        )
    return gm


def register_rmsnorm_bf16_fp8_quant_pass() -> None:
    register_dsv4_fusion_pass(
        "rmsnorm_bf16_fp8_quant_fx",
        apply_rmsnorm_bf16_fp8_quant_fx_pass,
        priority=10,
        env_gate="DSV4_FUSED_RMSNORM_BF16_FP8_QUANT",
    )


register_rmsnorm_bf16_fp8_quant_pass()
