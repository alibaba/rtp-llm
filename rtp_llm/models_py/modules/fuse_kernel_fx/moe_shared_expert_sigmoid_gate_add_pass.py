"""GraphFX replacement for MoE shared-expert sigmoid-gate add.

Eager unfused chain (after the DSV4-style strip in GenericMoeLayer):

    experts_output = experts_output + (torch.sigmoid(gate_output) * shared_expert_output)

This pass detects that chain and rewrites it to:

    sigmoid_gate_scale_add_triton(gate_output, shared_expert_output, experts_output)

which performs the (sigmoid + scale + add) in a single triton kernel, mutating
``experts_output`` in place.

Pattern anchor: ``add(experts, mul(sigmoid(gate), shared))``.

Note: ``sigmoid_gate_scale_add_triton`` is a mutating op; the rewrite uses it
as a side-effecting node and replaces uses of the outer ``add`` result with
the experts tensor (which now holds the fused result).
"""

from __future__ import annotations

import logging
import operator
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import target_name
from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    eliminate_dead_code_preserving_graphfx_side_effects,
    record_graphfx_fusion_hit,
    record_graphfx_fusion_miss,
    register_graphfx_fusion_pass,
)

try:
    from rtp_llm.models_py.triton_kernels.common.moe_gating import (
        sigmoid_gate_scale_add_triton,
    )
except Exception:  # noqa: BLE001

    def sigmoid_gate_scale_add_triton(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("sigmoid_gate_scale_add_triton unavailable")


logger = logging.getLogger(__name__)

_PASS_NAME = "moe_shared_expert_sigmoid_gate_add_fx"


# ----- predicates ----------------------------------------------------------


def _is_add(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and (
        node.target is operator.add
        or "aten.add" in str(node.target)
        or target_name(node.target) == "add"
    ):
        return True
    if node.op == "call_method" and target_name(node.target) in ("add", "add_"):
        return True
    return False


def _is_mul(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and (
        node.target is operator.mul
        or "aten.mul" in str(node.target)
        or target_name(node.target) == "mul"
    ):
        return True
    if node.op == "call_method" and target_name(node.target) in ("mul", "mul_"):
        return True
    return False


def _is_sigmoid(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function":
        s = str(node.target)
        if target_name(node.target) == "sigmoid" or "aten.sigmoid" in s:
            return True
    if node.op == "call_method" and target_name(node.target) == "sigmoid":
        return True
    return False


# ----- rewrite -------------------------------------------------------------


def _decompose_mul(
    mul_node: torch.fx.Node,
) -> Optional[Tuple[torch.fx.Node, torch.fx.Node]]:
    """Return (sigmoid_arg_of_mul, shared_arg_of_mul) — order normalised so
    sigmoid is always first.  Or None if the multiplication is not a
    sigmoid * shared shape.
    """
    if len(mul_node.args) != 2:
        return None
    a, b = mul_node.args
    if isinstance(a, torch.fx.Node) and _is_sigmoid(a):
        return a, b if isinstance(b, torch.fx.Node) else None  # type: ignore[return-value]
    if isinstance(b, torch.fx.Node) and _is_sigmoid(b):
        return b, a if isinstance(a, torch.fx.Node) else None  # type: ignore[return-value]
    return None


def _try_rewrite(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_add(node):
        return False
    if len(node.args) != 2:
        return False
    lhs, rhs = node.args
    if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
        return False

    # The mul (sigmoid(gate) * shared) may be on either side of the add.
    mul_node: Optional[torch.fx.Node] = None
    experts_node: Optional[torch.fx.Node] = None
    if _is_mul(rhs):
        mul_node, experts_node = rhs, lhs
    elif _is_mul(lhs):
        mul_node, experts_node = lhs, rhs
    else:
        return False

    decomp = _decompose_mul(mul_node)
    if decomp is None:
        record_graphfx_fusion_miss(_PASS_NAME, "mul_not_sigmoid_times_shared")
        return False
    sigmoid_node, shared_node = decomp
    if not isinstance(shared_node, torch.fx.Node):
        return False
    if not sigmoid_node.args:
        return False
    gate_node = sigmoid_node.args[0]
    if not isinstance(gate_node, torch.fx.Node):
        return False

    # Rewrite: insert sigmoid_gate_scale_add_triton(gate, shared, experts)
    # which mutates experts in place. Then replace uses of the add result
    # with experts_node (which now holds the fused output).
    with gm.graph.inserting_before(node):
        gm.graph.call_function(
            sigmoid_gate_scale_add_triton,
            args=(gate_node, shared_node, experts_node),
        )
    node.replace_all_uses_with(experts_node)
    return True


def apply_moe_shared_expert_sigmoid_gate_add_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    replaced = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            replaced += 1
    if replaced:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)
        record_graphfx_fusion_hit(_PASS_NAME, replaced)
        logger.info("GraphFX moe_shared_expert_sigmoid_gate_add pass: %d", replaced)
    return gm


def register_moe_shared_expert_sigmoid_gate_add_pass() -> None:
    register_graphfx_fusion_pass(
        _PASS_NAME,
        apply_moe_shared_expert_sigmoid_gate_add_fx_pass,
        priority=18,
        env_gate="GRAPHFX_FUSED_MOE_SHARED_EXPERT_SIGMOID_GATE_ADD",
    )


register_moe_shared_expert_sigmoid_gate_add_pass()
