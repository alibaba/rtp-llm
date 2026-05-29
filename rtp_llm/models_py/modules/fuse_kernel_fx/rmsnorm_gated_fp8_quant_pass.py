"""GraphFX replacement for Qwen3.5 RmsNormGated -> FP8 quant.

Eager pattern (qwen3_next.py:824-832 unfused branch):

    attn_output = self.norm(attn_out_reshaped, z_reshaped)   # RmsNormGated
    attn_output = attn_output.reshape(-1, num_heads * head_v_dim)
    fp8, scale = sgl_per_token_group_quant_fp8(attn_output, group_size=128, ...)

Replacement:

    fp8, scale = fused_rmsnorm_gated_fp8_quant(
        attn_out_reshaped, gate, weight, eps, num_heads,
        quant_group_size=128, scale_ue8m0=...
    )

The pass matches the lowered ``layer_norm_fwd(x, weight, bias, eps, z=gate, ...,
is_rms_norm=True, norm_before_gate=True)`` followed (through layout-only
views) by ``sgl_per_token_group_quant_fp8``.
"""

from __future__ import annotations

import logging
import operator

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
    all_users_are_layout_or_output_only,
    first_quant_consumer_of,
    is_call_function,
    is_getitem,
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
    from rtp_llm.models_py.triton_kernels.common.fused_rmsnorm_gated_fp8_quant import (
        fused_rmsnorm_gated_fp8_quant,
    )
except Exception:  # noqa: BLE001

    def fused_rmsnorm_gated_fp8_quant(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "fused_rmsnorm_gated_fp8_quant unavailable: triton/CUDA build required"
        )


logger = logging.getLogger(__name__)


def _is_layer_norm_fwd_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    return target_name(node.target) == "layer_norm_fwd"


def _is_rms_norm_gated(node: torch.fx.Node) -> bool:
    """Check that layer_norm_fwd was invoked in the RmsNormGated configuration."""
    kwargs = dict(node.kwargs)
    if not kwargs.get("is_rms_norm", False):
        return False
    if not kwargs.get("norm_before_gate", False):
        return False
    if kwargs.get("z", None) is None:
        return False
    return True


def _try_rewrite(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _is_layer_norm_fwd_node(node):
        return False
    if not _is_rms_norm_gated(node):
        return False
    args = list(node.args)
    if len(args) < 2:
        record_graphfx_fusion_miss(
            "rmsnorm_gated_fp8_quant_fx", "layer_norm_fwd_args_unparsed"
        )
        return False
    x_node = args[0]
    weight_node = args[1]
    eps = args[3] if len(args) > 3 else node.kwargs.get("eps", 1e-6)
    gate_node = node.kwargs["z"]
    group_size = node.kwargs.get("group_size", None)
    if group_size is None:
        last_dim = static_last_dim(x_node)
        if last_dim is None:
            record_graphfx_fusion_miss(
                "rmsnorm_gated_fp8_quant_fx", "unknown_group_size_and_last_dim"
            )
            return False
        # Treat the head_v_dim as the inner group when not explicit (matches
        # RmsNormGated default ``group_size = weight.shape[-1]``).
        group_size = last_dim
    # ``layer_norm_fwd`` returns ``(out, mean, rstd)``; the BF16 normed tensor
    # is reached through ``operator.getitem(layer_norm_fwd, 0)``.  Walk through
    # the index-0 getitem before searching for a quant consumer.
    bf16_node = None
    for user in list(node.users):
        if is_getitem(user, node, 0):
            bf16_node = user
            break
    if bf16_node is None:
        # Some custom RmsNormGated wrappers return a single tensor directly.
        bf16_node = node
    quant_node = first_quant_consumer_of(bf16_node)
    if quant_node is None:
        record_graphfx_fusion_miss(
            "rmsnorm_gated_fp8_quant_fx", "no_same_graph_quant_consumer"
        )
        return False
    if not quant_contract_ok(quant_node, group_size=128):
        record_graphfx_fusion_miss(
            "rmsnorm_gated_fp8_quant_fx", "quant_contract_mismatch"
        )
        return False
    # We need num_heads for the fused kernel.  At graph construction time we
    # don't have this in the FX layer_norm_fwd metadata; the model code passes
    # the raw value as a positional kwarg through the eager fast-path.  Rather
    # than guess, only fire when ``num_heads`` is present in the original call
    # (which is the case for the qwen3_next code path that explicitly threads
    # it through).  Otherwise miss and let eager run.
    num_heads = node.kwargs.get("num_heads", None)
    if num_heads is None:
        record_graphfx_fusion_miss(
            "rmsnorm_gated_fp8_quant_fx", "num_heads_not_threaded"
        )
        return False
    scale_ue8m0 = bool(quant_node.kwargs.get("scale_ue8m0", False))
    quant_group_size = quant_node.kwargs.get("group_size", 128)

    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            fused_rmsnorm_gated_fp8_quant,
            args=(x_node, gate_node, weight_node, float(eps), int(num_heads)),
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


# ---- cross-graph producer token ------------------------------------------


def graphfx_rmsnorm_gated_producer_token(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    scale_ue8m0: bool = False,
) -> torch.Tensor:
    """Producer token: compute RmsNormGated + quant, store (fp8, scale), return bf16."""
    from flash_attn.ops.triton.layer_norm import layer_norm_fwd

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import remember_quant

    result, _, _ = layer_norm_fwd(
        x,
        weight,
        None,
        eps,
        z=gate,
        is_rms_norm=True,
        norm_before_gate=True,
    )
    fp8, scale = sgl_per_token_group_quant_fp8(
        result,
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    remember_quant(result, fp8, scale)
    return result


def _rewrite_cross_graph_producer(
    gm: torch.fx.GraphModule, node: torch.fx.Node
) -> bool:
    """Replace RmsNormGated whose bf16 output only leaves through graph output."""
    if not _is_layer_norm_fwd_node(node):
        return False
    if not _is_rms_norm_gated(node):
        return False
    args = list(node.args)
    if len(args) < 2:
        return False
    x_node = args[0]
    weight_node = args[1]
    eps = args[3] if len(args) > 3 else node.kwargs.get("eps", 1e-6)
    gate_node = node.kwargs.get("z")
    if gate_node is None:
        return False
    bf16_node = None
    for user in list(node.users):
        if is_getitem(user, node, 0):
            bf16_node = user
            break
    if bf16_node is None:
        bf16_node = node
    if not all_users_are_layout_or_output_only(bf16_node, skip=set()):
        return False

    with gm.graph.inserting_before(node):
        token = gm.graph.call_function(
            graphfx_rmsnorm_gated_producer_token,
            args=(x_node, gate_node, weight_node),
            kwargs={"eps": float(eps), "scale_ue8m0": False},
        )
    for user in list(bf16_node.users):
        if user is token:
            continue
        user.replace_input_with(bf16_node, token)
    if bf16_node is not node and not bf16_node.users:
        gm.graph.erase_node(bf16_node)
    if not node.users:
        gm.graph.erase_node(node)
    return True


# ---- pass entry point -----------------------------------------------------


def apply_rmsnorm_gated_fp8_quant_fx_pass(
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
        if _is_layer_norm_fwd_node(node) and _rewrite_cross_graph_producer(gm, node):
            producer += 1
    if producer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    total = same_graph + producer
    if total:
        record_graphfx_fusion_hit("rmsnorm_gated_fp8_quant_fx", total)
        logger.info(
            "GraphFX rmsnorm_gated+FP8 quant pass: same_graph=%d producer=%d",
            same_graph,
            producer,
        )
    return gm


def register_rmsnorm_gated_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        "rmsnorm_gated_fp8_quant_fx",
        apply_rmsnorm_gated_fp8_quant_fx_pass,
        priority=15,
        env_gate="GRAPHFX_FUSED_RMSNORM_GATED_FP8_QUANT",
    )


register_rmsnorm_gated_fp8_quant_pass()
