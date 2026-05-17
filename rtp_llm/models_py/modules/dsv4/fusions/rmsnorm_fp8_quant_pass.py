"""GraphFX replacements for DSV4 RMSNorm -> FP8 quant.

Scenario
========
DSV4 Path1 feeds FP8 DeepGEMM through a normal RMSNorm output.  The unfused
same-graph form materializes BF16 RMSNorm and immediately quantizes it:

    x --------+
             v
    weight -> rmsnorm(x, weight, norm_eps) -> y_bf16
                                                |
                                                v
                       sgl_per_token_group_quant_fp8(y_bf16)
                                                |
                                                v
                                      (x_fp8, x_scale) -> DeepGEMM

When the BF16 RMSNorm value has no real consumer other than quant, this pass
replaces both launches with one fused CUDA wrapper:

    x --------+
             +--> fused_rmsnorm_fp8_quant(x, weight)
    weight --+                         |
                                      v
                             (x_fp8, x_scale) -> DeepGEMM

The compiled forward can also split producer and consumer into different FX
graphs.  In that case the producer graph records provenance and the consumer
graph resolves it:

    Graph A before: rmsnorm(x, weight, norm_eps) -> graph output
    Graph A after : dsv4_rmsnorm_quant_producer_token(x, weight, norm_eps)

    Graph B before: sgl_per_token_group_quant_fp8(graph_input_y)
    Graph B after : dsv4_fused_rmsnorm_fp8_quant_from_provenance(graph_input_y)

If ``DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8=1`` is enabled, the producer token can
run the BF16+FP8 fused kernel and stash precomputed FP8/scale tensors for the
consumer.  Otherwise the consumer uses provenance to recompute the fused
RMSNorm+quant path or falls back to the original quant when provenance is not
available.

Code key paths
==============
GraphFX install/registry:
``rtp_llm/models_py/modules/dsv4/fusions/graphfx_injector.py`` and
``rtp_llm/models_py/modules/dsv4/fusions/fusion_registry.py``.

Runtime/provenance bridge:
``rtp_llm/models_py/modules/dsv4/fusions/rmsnorm_quant_runtime.py``.

CUDA wrapper:
``rtp_llm/models_py/kernels/cuda/fused_rmsnorm_fp8_quant``.

Concrete DSV4 origins as of 2026-05-18
======================================
Normal block RMSNorm producers:
``rtp_llm/models_py/modules/dsv4/block.py:139-140`` constructs
``attn_norm`` and ``ffn_norm``.  Decode applies them at
``block.py:216`` and ``block.py:233``; prefill applies them at
``block.py:289`` and ``block.py:387``.  The module implementation is
``rtp_llm/models_py/modules/base/cuda/norm.py:50`` and
``norm.py:78``.

Immediate FP8 quant consumers:
``rtp_llm/models_py/modules/factory/linear/impl/cuda/fp8_deepgemm_linear.py:215``
quantizes BF16 activations with
``sgl_per_token_group_quant_fp8(group_size=128, eps=1e-4,
column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)``
before DeepGEMM.  DSV4 attention/indexer paths that feed this contract include
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:3884-3890`` and
``rtp_llm/models_py/modules/dsv4/fp8/indexer.py:279,378-386``.

Keep this pass limited to that DeepGEMM quant contract.  Cross-graph rewrites
are provenance bridges; without ``DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8=1`` they
may recompute RMSNorm in the consumer graph and should not be counted as a
launch-elimination path.
"""

from __future__ import annotations

import logging
import operator
import os

import torch

from rtp_llm.models_py.kernels.cuda.fused_rmsnorm_fp8_quant import fused_rmsnorm_fp8_quant
from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import (
    eliminate_dead_code_preserving_dsv4_side_effects,
    record_dsv4_fusion_hit,
    record_dsv4_fusion_miss,
    register_dsv4_fusion_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_quant_runtime import (
    dsv4_fused_rmsnorm_fp8_quant_from_provenance,
    dsv4_rmsnorm_quant_mutating_producer_token,
    dsv4_rmsnorm_quant_producer_token,
)

logger = logging.getLogger(__name__)

_SUPPORTED_HIDDEN_DIMS = {1024, 1536, 2048, 3072, 4096, 7168}
_PRODUCER_SCOPE_DENY_SUBSTRINGS = (
    "final_ln",
    "final_norm",
    "input_layernorm",
    "post_attention_layernorm",
)


def dsv4_rmsnorm_fx_ref(x: torch.Tensor, weight: torch.Tensor, eps: float):
    y = x.float()
    inv_rms = torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
    return (y * inv_rms * weight.float()).to(torch.bfloat16)


def _target_name(target) -> str:
    return getattr(target, "__name__", str(target))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _is_quant_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and _target_name(node.target) == "sgl_per_token_group_quant_fp8"


def _is_plain_rmsnorm_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = _target_name(node.target).lower()
    return "rmsnorm" in name and "rope" not in name and "quant" not in name and "add" not in name


def _unwrap_contiguous(node: object) -> object:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_method"
        and _target_name(node.target) == "contiguous"
        and len(node.args) >= 1
    ):
        return node.args[0]
    return node


def _unwrap_layout_only(node: object) -> object:
    while isinstance(node, torch.fx.Node):
        if node.op == "call_method" and _target_name(node.target) in (
            "contiguous",
            "reshape",
            "view",
        ):
            node = node.args[0] if node.args else node
            continue
        if node.op == "call_function" and _target_name(node.target) in (
            "reshape",
            "view",
        ):
            node = node.args[0] if node.args else node
            continue
        return node
    return node


def _quant_contract_ok(node: torch.fx.Node) -> bool:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    group_size = kwargs.get("group_size", args[1] if len(args) > 1 else None)
    eps = kwargs.get("eps", 1e-10)
    return (
        group_size == 128
        and bool(kwargs.get("column_major_scales", False))
        and bool(kwargs.get("scale_tma_aligned", False))
        and bool(kwargs.get("scale_ue8m0", False))
        and not bool(kwargs.get("fuse_silu_and_mul", False))
        and kwargs.get("masked_m", None) is None
        and eps is not None
    )


def _rmsnorm_args(node: torch.fx.Node):
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) >= 3:
        return args[0], args[1], args[2]
    return args[0], args[1], kwargs.get("eps", 1e-6)


def _is_getitem(node: object, source: torch.fx.Node, index: int) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and node.args[0] is source
        and node.args[1] == index
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


def _static_shape(node: object):
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_meta = node.meta.get("tensor_meta")
    return getattr(tensor_meta, "shape", None)


def _static_last_dim(node: object):
    shape = _static_shape(node)
    if not shape:
        return None
    value = shape[-1]
    return value if isinstance(value, int) else None


def _static_numel_1d(node: object):
    shape = _static_shape(node)
    if not shape or len(shape) != 1:
        return None
    value = shape[0]
    return value if isinstance(value, int) else None


def _hidden_contract_ok(x: object, weight: object) -> bool:
    # Only fixed hidden/weight dimensions participate in FX matching. Dynamic
    # M/bs/seq dimensions are deliberately ignored and are validated by the
    # CUDA wrapper at runtime through shape/stride/dtype checks.
    k = _static_last_dim(x)
    if k is not None and k not in _SUPPORTED_HIDDEN_DIMS:
        return False
    weight_k = _static_numel_1d(weight)
    if weight_k is not None and weight_k not in _SUPPORTED_HIDDEN_DIMS:
        return False
    if k is not None and weight_k is not None and k != weight_k:
        return False
    return True


def _node_label(node: object, seen: set[torch.fx.Node] | None = None) -> str:
    if not isinstance(node, torch.fx.Node):
        return str(node)
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


def _producer_scope_ok(x: object, weight: object) -> bool:
    """Keep producer tokens on RMSNorm sites that can feed FP8 linear quant.

    The DSV4 GraphFX boundary is intentionally above individual layers, but
    Dynamo still splits pybind RMSNorm and FP8 linear calls into separate FX
    segments.  Some useful producer graphs retain module-qualified names such
    as ``attn_norm``/``ffn_norm`` while q_lora local helper graphs often only
    expose generic placeholders such as ``L_self_weight`` or ``L_weight_``.
    Matching therefore rejects only known non-linear-output norms and relies on
    fixed hidden-dim matching plus runtime dtype/shape/stride validation for
    safety.
    """
    text = f"{_node_label(x)} {_node_label(weight)}"
    if any(token in text for token in _PRODUCER_SCOPE_DENY_SUBSTRINGS):
        record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "producer_scope_denied")
        return False
    padded = f" {text} "
    if (
        "l_weight_" not in text
        and "self_weight" not in text
        and " weight " not in padded
        and "_weight" not in text
    ):
        record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "producer_scope_not_attention_local")
        return False
    return True


def _replace_quant_uses(quant_node: torch.fx.Node, q_node: torch.fx.Node, s_node: torch.fx.Node) -> None:
    for user in list(quant_node.users):
        if _is_getitem(user, quant_node, 0):
            user.replace_all_uses_with(q_node)
        elif _is_getitem(user, quant_node, 1):
            user.replace_all_uses_with(s_node)
        else:
            raise RuntimeError(
                "DSV4 RMSNorm+FP8 quant pass expects quant tuple users "
                f"to be operator.getitem(0/1), got target={_target_name(user.target)}"
            )


def _erase_dead_nodes(nodes: set[torch.fx.Node]) -> None:
    for node in list(nodes):
        if len(node.users) == 0:
            node.graph.erase_node(node)


def _graph_output_users(node: torch.fx.Node) -> list[torch.fx.Node]:
    return [user for user in list(node.users) if user.op == "output"]


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
    node: torch.fx.Node, seen: set[torch.fx.Node] | None = None
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


def _insert_functional_rmsnorm_token(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_plain_rmsnorm_node(node) or len(node.users) == 0:
        return False
    x, weight, norm_eps = _rmsnorm_args(node)
    if not _hidden_contract_ok(x, weight):
        record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "producer_unsupported_fixed_hidden_dim")
        return False
    if not _producer_scope_ok(x, weight):
        return False
    if not _all_users_are_layout_outputs(node):
        return False
    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(
            dsv4_rmsnorm_quant_producer_token,
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
        record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "producer_unsupported_fixed_hidden_dim")
        return False
    if not _producer_scope_ok(x, weight):
        return False
    old_users = [user for user in _graph_output_users(output_node) if user is not node]
    if not old_users:
        with gm.graph.inserting_before(node):
            gm.graph.call_function(
                dsv4_rmsnorm_quant_mutating_producer_token,
                args=(output_node, x, weight, norm_eps),
            )
        if len(node.users) == 0:
            gm.graph.erase_node(node)
        return True
    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(
            dsv4_rmsnorm_quant_producer_token,
            args=(x, weight, norm_eps),
        )
    for user in old_users:
        user.replace_input_with(output_node, token)
    if len(node.users) == 0:
        gm.graph.erase_node(node)
    if len(output_node.users) == 0:
        gm.graph.erase_node(output_node)
    return True


def _insert_rmsnorm_provenance_tokens(gm: torch.fx.GraphModule) -> int:
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
        record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "cross_graph_quant_contract_mismatch")
        return False
    x_node = _unwrap_layout_only(node.args[0] if node.args else None)
    if _match_rmsnorm_value(x_node) is not None:
        return False
    quant_eps = node.kwargs.get("eps", 1e-10)
    provenance_input = x_node if isinstance(x_node, torch.fx.Node) else node.args[0]
    fallback_input = node.args[0] if node.args else None
    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            dsv4_fused_rmsnorm_fp8_quant_from_provenance,
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
    if _env_flag("DSV4_RMSNORM_QUANT_DEBUG") or _env_flag("DSV4_FUSION_REGISTRY_DEBUG"):
        logger.info(
            "DSV4 RMSNorm+FP8 quant cross-graph consumer rewrite: input=%s fallback=%s",
            _node_label(provenance_input),
            _node_label(fallback_input),
        )
    node.replace_all_uses_with(fused)
    return True


def apply_rmsnorm_fp8_quant_fx_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    replaced = 0
    cross_graph_consumers = 0
    for node in list(gm.graph.nodes):
        if not _is_quant_node(node):
            continue
        if not _quant_contract_ok(node):
            record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "quant_contract_mismatch")
            continue
        x_node = _unwrap_layout_only(node.args[0] if node.args else None)
        rmsnorm = _match_rmsnorm_value(x_node)
        if rmsnorm is None:
            record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "quant_input_not_rmsnorm")
            continue
        if (
            isinstance(x_node, torch.fx.Node)
            and rmsnorm["skip_users"]
            and _has_bf16_consumer(x_node, node, rmsnorm["skip_users"])
        ):
            record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "bf16_consumer_requires_dual_output")
            continue
        x, weight, norm_eps = rmsnorm["x"], rmsnorm["weight"], rmsnorm["norm_eps"]
        if not _hidden_contract_ok(x, weight):
            record_dsv4_fusion_miss("rmsnorm_fp8_quant_fx", "unsupported_fixed_hidden_dim")
            continue
        quant_eps = node.kwargs.get("eps", 1e-10)
        with gm.graph.inserting_before(rmsnorm["insert_before"]):
            fused = gm.graph.call_function(
                fused_rmsnorm_fp8_quant,
                args=(x, weight),
                kwargs={"norm_eps": norm_eps, "quant_eps": quant_eps, "group_size": 128},
            )
        node.replace_all_uses_with(fused)
        _erase_dead_nodes(rmsnorm["skip_users"])
        replaced += 1
    if replaced:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
    for node in list(gm.graph.nodes):
        if _rewrite_quant_from_provenance(gm, node):
            cross_graph_consumers += 1
    provenance_tokens = _insert_rmsnorm_provenance_tokens(gm)
    if replaced or cross_graph_consumers or provenance_tokens:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
        record_dsv4_fusion_hit(
            "rmsnorm_fp8_quant_fx",
            replaced + cross_graph_consumers + provenance_tokens,
        )
        logger.info(
            "DSV4 FX RMSNorm+FP8 quant pass: same_graph=%d cross_graph_consumers=%d "
            "producer_tokens=%d",
            replaced,
            cross_graph_consumers,
            provenance_tokens,
        )
    return gm


def register_rmsnorm_fp8_quant_pass() -> None:
    register_dsv4_fusion_pass(
        "rmsnorm_fp8_quant_fx",
        apply_rmsnorm_fp8_quant_fx_pass,
        priority=11,
        env_gate="DSV4_FUSED_RMSNORM_FP8_QUANT",
    )


register_rmsnorm_fp8_quant_pass()
