"""GraphFX replacement for ``attn_output * sigmoid(gate) -> FP8 quant`` (Qwen3.5).

Eager unfused chain in CausalAttention (when ``_fuse_sigmoid_mul_quant`` is False):

    attn_output = self.sigmoid_mul(attn_output, gate)   # SigmoidMulInplace
    output = self.o_proj(attn_output)                   # CudaFp8GEMMLinear

When ``o_proj`` is FP8 DeepGEMM, the visible quant is
``sgl_per_token_group_quant_fp8(attn_output, group_size=128, ...)``.  This pass
replaces the chain with ``sigmoid_mul_fp8_quant_fwd`` whenever the fused
kernel's contract holds (last-dim divisible by 128).
"""

from __future__ import annotations

import logging
import operator

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
    first_quant_consumer_of,
    quant_contract_ok,
    replace_quant_uses,
    static_last_dim,
    target_name,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    eliminate_dead_code_preserving_graphfx_side_effects,
    record_graphfx_fusion_hit,
    record_graphfx_fusion_miss,
    register_graphfx_fusion_pass,
)

try:
    from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
        sigmoid_mul_fp8_quant_fwd,
    )
except Exception:  # noqa: BLE001

    def sigmoid_mul_fp8_quant_fwd(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("sigmoid_mul_fp8_quant_fwd unavailable")


logger = logging.getLogger(__name__)


_SIGMOID_MUL_TRITON_TARGETS = {"sigmoid_mul_inplace_triton", "SigmoidMulInplace"}


def _is_sigmoid_mul_triton_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    return target_name(node.target) in _SIGMOID_MUL_TRITON_TARGETS


def _is_sigmoid_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and target_name(node.target) == "sigmoid":
        return True
    if node.op == "call_method" and target_name(node.target) == "sigmoid":
        return True
    return False


def _is_mul_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and node.target in (operator.mul, torch.mul):
        return True
    if node.op == "call_method" and target_name(node.target) == "mul":
        return True
    return False


def _decompose_mul_sigmoid(node: torch.fx.Node):
    """If ``node`` is ``attn * sigmoid(gate)``, return (attn, gate); else None.

    Accepts either operand order; ``sigmoid`` may be ``torch.sigmoid`` (call_function)
    or ``.sigmoid()`` (call_method).
    """
    if not _is_mul_node(node) or len(node.args) < 2:
        return None
    a, b = node.args[0], node.args[1]
    if _is_sigmoid_node(b) and isinstance(a, torch.fx.Node) and b.args:
        return a, b.args[0]
    if _is_sigmoid_node(a) and isinstance(b, torch.fx.Node) and a.args:
        return b, a.args[0]
    return None


def _producer_inputs(node: torch.fx.Node):
    """Return ``(attn_node, gate_node)`` from either producer pattern, else None.

    Producer pattern is either:
      * ``sigmoid_mul_inplace_triton(attn, gate)`` (eager triton baseline), OR
      * ``attn * torch.sigmoid(gate)`` (pure-PyTorch baseline that becomes the
        default once ``causal_attention`` strips its ``_fuse_sigmoid_mul_quant``
        fast path under the DSV4-style refactor).
    """
    if _is_sigmoid_mul_triton_node(node):
        if len(node.args) < 2:
            return None
        return node.args[0], node.args[1]
    return _decompose_mul_sigmoid(node)


def _try_rewrite(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    inputs = _producer_inputs(node)
    if inputs is None:
        return False
    attn_node, gate_node = inputs
    last_dim = static_last_dim(attn_node)
    if last_dim is not None and last_dim % 128 != 0:
        record_graphfx_fusion_miss(
            "sigmoid_mul_fp8_quant_fx", "last_dim_not_divisible_by_128"
        )
        return False
    quant_node = first_quant_consumer_of(node)
    if quant_node is None:
        record_graphfx_fusion_miss(
            "sigmoid_mul_fp8_quant_fx", "no_same_graph_quant_consumer"
        )
        return False
    if not quant_contract_ok(quant_node, group_size=128):
        record_graphfx_fusion_miss(
            "sigmoid_mul_fp8_quant_fx", "quant_contract_mismatch"
        )
        return False
    scale_ue8m0 = bool(quant_node.kwargs.get("scale_ue8m0", False))
    quant_group_size = quant_node.kwargs.get("group_size", 128)

    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            sigmoid_mul_fp8_quant_fwd,
            args=(attn_node, gate_node),
            kwargs={
                "quant_group_size": int(quant_group_size),
                "scale_ue8m0": scale_ue8m0,
            },
        )
        q_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
        s_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
    replace_quant_uses(quant_node, q_node, s_node)
    if not node.users:
        gm.graph.erase_node(node)
    return True


def apply_sigmoid_mul_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    replaced = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            replaced += 1
    if replaced:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)
        record_graphfx_fusion_hit("sigmoid_mul_fp8_quant_fx", replaced)
        logger.info("GraphFX sigmoid_mul+FP8 quant pass: %d", replaced)
    return gm


def register_sigmoid_mul_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        "sigmoid_mul_fp8_quant_fx",
        apply_sigmoid_mul_fp8_quant_fx_pass,
        priority=25,
        env_gate="GRAPHFX_FUSED_SIGMOID_MUL_FP8_QUANT",
    )


register_sigmoid_mul_fp8_quant_pass()
