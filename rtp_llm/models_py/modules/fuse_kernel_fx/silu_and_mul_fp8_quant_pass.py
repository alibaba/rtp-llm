"""GraphFX replacement for SiLU-and-mul -> FP8 quant.

Eager unfused chain in DenseMLP (when ``_fuse_silu_quant`` is False):

    activated = self.act_fn(up)            # silu_and_mul -> bf16 [T, H]
    output = self.down_proj(activated)     # CudaFp8GEMMLinear -> internal sgl quant

When the down_proj is an FP8 DeepGEMM linear, the quant inside expands to the
visible ``sgl_per_token_group_quant_fp8(activated, group_size=128, ...)``.
This pass replaces the chain with the dense fused kernel.
"""

from __future__ import annotations

import logging
import operator

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
    first_quant_consumer_of,
    is_call_function,
    quant_contract_ok,
    replace_quant_uses,
    static_last_dim,
    target_name,
    unwrap_layout_only,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    eliminate_dead_code_preserving_graphfx_side_effects,
    record_graphfx_fusion_hit,
    record_graphfx_fusion_miss,
    register_graphfx_fusion_pass,
)

try:
    from rtp_llm.models_py.triton_kernels.common.activation import (
        silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
    )
except Exception:  # noqa: BLE001

    def silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError(
            "silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd unavailable"
        )


logger = logging.getLogger(__name__)


_SILU_TARGET_NAMES = {"silu_and_mul", "FusedSiluAndMul"}


def _is_silu_and_mul_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = target_name(node.target)
    return name in _SILU_TARGET_NAMES


def _last_dim_divisible_by(node: object, divisor: int) -> bool:
    last = static_last_dim(node)
    if last is None:
        # Trust runtime check inside the kernel wrapper for dynamic shapes.
        return True
    return last % divisor == 0


def _try_rewrite(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_silu_and_mul_node(node):
        return False
    if not node.args:
        record_graphfx_fusion_miss("silu_and_mul_fp8_quant_fx", "no_args")
        return False
    up_node = node.args[0]
    if not _last_dim_divisible_by(up_node, 256):
        # silu_and_mul halves the last dim; the result must still be divisible
        # by 128 (group_size).  256 == 2 * 128.
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "input_last_dim_not_divisible_by_256"
        )
        return False
    quant_node = first_quant_consumer_of(node)
    if quant_node is None:
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "no_same_graph_quant_consumer"
        )
        return False
    if not quant_contract_ok(quant_node, group_size=128):
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "quant_contract_mismatch"
        )
        return False
    scale_ue8m0 = bool(quant_node.kwargs.get("scale_ue8m0", False))
    quant_group_size = quant_node.kwargs.get("group_size", 128)

    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
            args=(up_node,),
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


def apply_silu_and_mul_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    replaced = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            replaced += 1
    if replaced:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)
        record_graphfx_fusion_hit("silu_and_mul_fp8_quant_fx", replaced)
        logger.info("GraphFX silu_and_mul+FP8 quant pass: %d", replaced)
    return gm


def register_silu_and_mul_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        "silu_and_mul_fp8_quant_fx",
        apply_silu_and_mul_fp8_quant_fx_pass,
        priority=20,
        env_gate="GRAPHFX_FUSED_SILU_AND_MUL_FP8_QUANT",
    )


register_silu_and_mul_fp8_quant_pass()
