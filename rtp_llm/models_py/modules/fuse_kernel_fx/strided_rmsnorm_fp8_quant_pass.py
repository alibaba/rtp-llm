"""GraphFX replacement for strided RMSNorm (MLA q_a / kv_a layernorm paths).

Eager unfused chain in MlaAttention.forward:

    # compressed_kv is a strided view from torch.split
    compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

Which expands (after RMSNorm.forward inline) to:

    contiguous_input = strided_view.contiguous()   # copy
    output = torch.empty_like(contiguous_input)
    rtp_llm_graphfx::rmsnorm_mutating(output, contiguous_input, weight, eps, stream)
    # output is used downstream

This pass replaces the chain with:
    fused_strided_rmsnorm(strided_view, weight, eps)

For the fp8 variant (when a quant consumer follows), it replaces with:
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(strided_view, weight, eps, ...)
"""

from __future__ import annotations

import logging
import operator
from typing import Optional

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
    first_quant_consumer_of,
    is_call_function,
    is_call_method,
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
    from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
        fused_strided_rmsnorm,
        fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
    )
except Exception:  # noqa: BLE001

    def fused_strided_rmsnorm(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("fused_strided_rmsnorm unavailable")

    def fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError(
            "fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output unavailable"
        )


logger = logging.getLogger(__name__)

_PASS_NAME = "strided_rmsnorm_fp8_quant_fx"

_RMSNORM_TARGET_NAMES = {"rmsnorm_mutating", "rmsnorm"}


def _is_rmsnorm_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = target_name(node.target)
    return name in _RMSNORM_TARGET_NAMES


def _input_from_contiguous(rmsnorm_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the pre-.contiguous() source if rmsnorm input came from contiguous().

    RMSNorm.forward pattern: rmsnorm(output, input, weight, eps, stream)
    where input = some_tensor.contiguous()
    """
    if len(rmsnorm_node.args) < 2:
        return None
    input_node = rmsnorm_node.args[1]
    if not isinstance(input_node, torch.fx.Node):
        return None
    if is_call_method(input_node, "contiguous"):
        if input_node.args:
            pre = input_node.args[0]
            if isinstance(pre, torch.fx.Node):
                return pre
    return None


def _find_output_node(rmsnorm_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the output tensor node (arg[0] of rmsnorm, which is also returned).

    Pattern: output = empty_like(input); rmsnorm(output, input, w, e, s); ... use output ...
    """
    if len(rmsnorm_node.args) < 1:
        return None
    output_node = rmsnorm_node.args[0]
    if isinstance(output_node, torch.fx.Node):
        return output_node
    return None


def _is_from_split(node: torch.fx.Node) -> bool:
    """Check if node is likely a strided view from split/getitem."""
    if node.op == "call_function" and node.target == operator.getitem:
        if node.args and isinstance(node.args[0], torch.fx.Node):
            src = node.args[0]
            src_name = target_name(src.target) if src.op == "call_function" else ""
            if "split" in src_name:
                return True
    return False


def _try_rewrite_bf16(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite contiguous+rmsnorm to fused_strided_rmsnorm (bf16 output only)."""
    if not _is_rmsnorm_node(node):
        return False
    pre_contiguous = _input_from_contiguous(node)
    if pre_contiguous is None:
        return False
    if not _is_from_split(pre_contiguous):
        record_graphfx_fusion_miss(_PASS_NAME, "input_not_from_split")
        return False
    output_node = _find_output_node(node)
    if output_node is None:
        record_graphfx_fusion_miss(_PASS_NAME, "no_output_node")
        return False
    if len(node.args) < 4:
        record_graphfx_fusion_miss(_PASS_NAME, "insufficient_args")
        return False
    weight_node = node.args[2]
    eps_node = node.args[3]

    quant_node = first_quant_consumer_of(output_node)
    if quant_node is not None and quant_contract_ok(quant_node, group_size=128):
        return _try_rewrite_fp8(gm, node, pre_contiguous, output_node, weight_node, eps_node, quant_node)

    with gm.graph.inserting_before(node):
        fused_result = gm.graph.call_function(
            fused_strided_rmsnorm,
            args=(pre_contiguous, weight_node, eps_node),
        )
    output_node.replace_all_uses_with(fused_result)
    return True


def _try_rewrite_fp8(
    gm: torch.fx.GraphModule,
    rmsnorm_node: torch.fx.Node,
    pre_contiguous: torch.fx.Node,
    output_node: torch.fx.Node,
    weight_node: object,
    eps_node: object,
    quant_node: torch.fx.Node,
) -> bool:
    """Rewrite contiguous+rmsnorm+quant to fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output."""
    scale_ue8m0 = bool(quant_node.kwargs.get("scale_ue8m0", False))
    quant_group_size = quant_node.kwargs.get("group_size", 128)

    with gm.graph.inserting_before(rmsnorm_node):
        fused = gm.graph.call_function(
            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
            args=(pre_contiguous, weight_node, eps_node),
            kwargs={
                "group_size": int(quant_group_size),
                "scale_ue8m0": scale_ue8m0,
            },
        )
        bf16_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
        fp8_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
        scale_node = gm.graph.call_function(operator.getitem, args=(fused, 2))

    output_node.replace_all_uses_with(bf16_node)
    replace_quant_uses(quant_node, fp8_node, scale_node)
    return True


def apply_strided_rmsnorm_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    replaced = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite_bf16(gm, node):
            replaced += 1
    if replaced:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)
        record_graphfx_fusion_hit(_PASS_NAME, replaced)
        logger.info("GraphFX strided_rmsnorm pass: %d", replaced)
    return gm


def register_strided_rmsnorm_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        _PASS_NAME,
        apply_strided_rmsnorm_fp8_quant_fx_pass,
        priority=15,
        env_gate="GRAPHFX_FUSED_STRIDED_RMSNORM_FP8_QUANT",
    )


register_strided_rmsnorm_fp8_quant_pass()
