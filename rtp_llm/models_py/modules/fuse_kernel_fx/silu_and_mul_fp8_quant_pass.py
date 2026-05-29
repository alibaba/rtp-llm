"""GraphFX replacement for SiLU-and-mul -> FP8 quant.

Eager unfused chain in DenseMLP (when ``_fuse_silu_quant`` is False):

    activated = self.act_fn(up)            # silu_and_mul -> bf16 [T, H]
    output = self.down_proj(activated)     # CudaFp8GEMMLinear -> internal sgl quant

When the down_proj is an FP8 DeepGEMM linear, the quant inside expands to the
visible ``sgl_per_token_group_quant_fp8(activated, group_size=128, ...)``.
This pass replaces the chain with the dense fused kernel.

Cross-graph support: when the quant lives in a different FX subgraph (inside
CudaFp8GEMMLinear.forward in eager mode after a graph break), the pass emits
a producer token that computes silu_and_mul + quant and stores (fp8, scale) in
the shared provenance registry.
"""

from __future__ import annotations

import logging
import operator

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
    all_users_are_layout_or_output_only,
    first_quant_consumer_of,
    is_call_function,
    quant_contract_ok,
    quant_group_size,
    quant_scale_ue8m0,
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


_SILU_TARGET_NAMES = {
    "silu_and_mul",
    "FusedSiluAndMul",
    "rtp_llm_graphfx::silu_and_mul",
}


def _is_silu_and_mul_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = target_name(node.target)
    if name in _SILU_TARGET_NAMES:
        return True
    # Dynamo may emit the custom_op OpOverload directly; the wrapper's
    # __name__ is "silu_and_mul" but the underlying op stringifies as
    # ``rtp_llm_graphfx.silu_and_mul.default``.
    qualified = str(node.target)
    return "silu_and_mul" in qualified


def _last_dim_divisible_by(node: object, divisor: int) -> bool:
    last = static_last_dim(node)
    if last is None:
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
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "input_last_dim_not_divisible_by_256"
        )
        return False
    quant_node = first_quant_consumer_of(node)
    if quant_node is None:
        logger.warning(
            "GraphFX silu_and_mul MISS: no quant consumer. node=%s users=%s",
            node.name,
            [u.name + ":" + target_name(u.target) for u in node.users],
        )
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "no_same_graph_quant_consumer"
        )
        return False
    if not quant_contract_ok(quant_node, group_size=128):
        record_graphfx_fusion_miss(
            "silu_and_mul_fp8_quant_fx", "quant_contract_mismatch"
        )
        return False
    scale_ue8m0 = quant_scale_ue8m0(quant_node)
    qgs = quant_group_size(quant_node)

    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
            args=(up_node,),
            kwargs={
                "quant_group_size": qgs,
                "scale_ue8m0": scale_ue8m0,
            },
        )
        q_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
        s_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
    replace_quant_uses(quant_node, q_node, s_node)
    if not node.users:
        gm.graph.erase_node(node)
    return True


# ---- cross-graph producer token ------------------------------------------


def graphfx_silu_and_mul_producer_token(
    up: torch.Tensor,
    *,
    scale_ue8m0: bool = False,
) -> torch.Tensor:
    """Producer token: run the FUSED silu_and_mul + fp8 quant kernel once and
    stash ``(fp8, scale)`` keyed on the bf16 output. Returns the bf16 tensor
    so downstream code (including a fallback quant on a cache miss) sees the
    real silu-and-mul result. Replaces the separate ``act_and_mul_kernel`` +
    standalone quant pair previously emitted by this producer.
    """
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.base import FusedSiluAndMul
    from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import remember_quant

    try:
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul_per_token_group_fp8_quant_dense_packed_with_bf16_fwd,
        )
    except Exception:  # noqa: BLE001 - fallback when triton/activation unavailable
        silu_and_mul_per_token_group_fp8_quant_dense_packed_with_bf16_fwd = None  # type: ignore[assignment]

    from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import (
        slot_push_quant,
    )

    # Fast path: one fused Triton launch produces bf16 + fp8 + scale.
    if (
        silu_and_mul_per_token_group_fp8_quant_dense_packed_with_bf16_fwd is not None
        and up.dim() == 2
        and up.is_contiguous()
        and up.shape[-1] % 256 == 0  # quant_group_size=128 * 2 (gate|up split)
    ):
        try:
            bf16_out, fp8, scale = (
                silu_and_mul_per_token_group_fp8_quant_dense_packed_with_bf16_fwd(
                    up,
                    quant_group_size=128,
                    scale_ue8m0=scale_ue8m0,
                )
            )
        except Exception:  # noqa: BLE001 - fall through to legacy path
            bf16_out = None  # type: ignore[assignment]
        else:
            remember_quant(bf16_out, fp8, scale)
            slot_push_quant(fp8, scale, tag="silu")
            return bf16_out

    # Legacy fallback: separate silu_and_mul + standalone quant (each is one
    # kernel launch — ``act_and_mul_kernel`` will appear in the timeline).
    result = FusedSiluAndMul()(up)
    fp8, scale = sgl_per_token_group_quant_fp8(
        result,
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    remember_quant(result, fp8, scale)
    slot_push_quant(fp8, scale, tag="silu")
    return result


def _rewrite_cross_graph_producer(
    gm: torch.fx.GraphModule, node: torch.fx.Node
) -> bool:
    """Replace silu_and_mul whose output only leaves through graph output."""
    if not _is_silu_and_mul_node(node):
        return False
    if not node.args:
        return False
    up_node = node.args[0]
    if not _last_dim_divisible_by(up_node, 256):
        return False
    if not all_users_are_layout_or_output_only(node, skip=set()):
        return False

    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(
            graphfx_silu_and_mul_producer_token,
            args=(up_node,),
            kwargs={"scale_ue8m0": True},
        )
    for user in list(node.users):
        if user is token:
            continue
        user.replace_input_with(node, token)
    if not node.users:
        gm.graph.erase_node(node)
    return True


# ---- pass entry point -----------------------------------------------------


def apply_silu_and_mul_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    same_graph = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            same_graph += 1
    if same_graph:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    producer = 0
    for node in list(gm.graph.nodes):
        if _is_silu_and_mul_node(node) and _rewrite_cross_graph_producer(gm, node):
            producer += 1
    if producer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    total = same_graph + producer
    if total:
        record_graphfx_fusion_hit("silu_and_mul_fp8_quant_fx", total)
        logger.info(
            "GraphFX silu_and_mul+FP8 quant pass: same_graph=%d producer=%d",
            same_graph,
            producer,
        )
    return gm


def register_silu_and_mul_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        "silu_and_mul_fp8_quant_fx",
        apply_silu_and_mul_fp8_quant_fx_pass,
        priority=20,
        env_gate="GRAPHFX_FUSED_SILU_AND_MUL_FP8_QUANT",
    )


register_silu_and_mul_fp8_quant_pass()
