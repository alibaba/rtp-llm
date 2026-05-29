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
    all_users_are_layout_or_output_only,
    first_quant_consumer_of,
    is_call_function,
    is_call_method,
    quant_contract_ok,
    quant_group_size,
    quant_scale_ue8m0,
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
    if name in _RMSNORM_TARGET_NAMES:
        return True
    # Dynamo may emit the OpOverload directly (target.__name__ is the overload
    # like "default"); fall back to qualified string form to catch it.
    # IMPORTANT: must NOT match ``fused_add_rmsnorm_mutating`` (handled by the
    # add_rmsnorm pass) even though that name contains ``rmsnorm_mutating`` as
    # a substring.  Use endswith on the qualified-name suffix.
    qualified = str(node.target)
    if "fused_add_rmsnorm" in qualified:
        return False
    return (
        qualified.endswith(".rmsnorm_mutating.default")
        or qualified.endswith("::rmsnorm_mutating")
        or qualified.endswith(".rmsnorm.default")
    )


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
    """Check if node is likely a strided view from split/getitem.

    Dynamo lowers ``torch.split`` to ``aten.split.Tensor`` / ``aten.split_with_sizes.default``
    whose ``__name__`` is just the overload (``"Tensor"`` / ``"default"``); it may also
    lower ``torch.split`` to plain ``getitem(tensor, slice_tuple)`` when one dim is sliced
    out (the MLA q_a / kv_a paths do this).  Both forms are accepted so the producer-side
    rewrite recognises them as candidates for ``fused_strided_rmsnorm``.
    """
    if node.op == "call_function" and node.target == operator.getitem:
        if node.args and isinstance(node.args[0], torch.fx.Node):
            src = node.args[0]
            if src.op == "call_function":
                if "split" in target_name(src.target) or "split" in str(src.target):
                    return True
        # Slice-tuple getitem: getitem(tensor, (slice, slice, slice(...))).
        # This is what Dynamo emits when ``torch.split`` is fused with slicing
        # in the same source line, e.g. ``compressed_kv[:, :, :kv_lora_rank]``.
        if len(node.args) >= 2:
            idx = node.args[1]
            if isinstance(idx, tuple) and any(isinstance(s, slice) for s in idx):
                return True
    return False


def _try_rewrite_bf16(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite contiguous+rmsnorm to fused_strided_rmsnorm (bf16 output only).

    Also handles the case where the producer pass already eliminated .contiguous()
    and the rmsnorm input is directly a strided split view.
    """
    if not _is_rmsnorm_node(node):
        return False
    pre_contiguous = _input_from_contiguous(node)
    if pre_contiguous is None:
        # Producer rewrite may have eliminated .contiguous() — check if the
        # rmsnorm input itself is directly from a split (strided view).
        if len(node.args) >= 2 and isinstance(node.args[1], torch.fx.Node):
            direct_input = node.args[1]
            if _is_from_split(direct_input):
                pre_contiguous = direct_input
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
        return _try_rewrite_fp8(
            gm, node, pre_contiguous, output_node, weight_node, eps_node, quant_node
        )

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
    scale_ue8m0 = quant_scale_ue8m0(quant_node)
    qgs = quant_group_size(quant_node)

    with gm.graph.inserting_before(rmsnorm_node):
        fused = gm.graph.call_function(
            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
            args=(pre_contiguous, weight_node, eps_node),
            kwargs={
                "group_size": qgs,
                "scale_ue8m0": scale_ue8m0,
            },
        )
        bf16_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
        fp8_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
        scale_node = gm.graph.call_function(operator.getitem, args=(fused, 2))

    output_node.replace_all_uses_with(bf16_node)
    replace_quant_uses(quant_node, fp8_node, scale_node)
    return True


# ---- cross-graph producer + consumer (handles RMSNorm.forward graph break) ----
#
# The eager chain is:
#
#     contig = strided_view.contiguous()            # in MLA outer subgraph
#     output = torch.empty_like(contig)             # in MLA outer subgraph
#     rtp_llm_ops.rmsnorm(output, contig, weight, eps, stream_id)  # in its own subgraph
#                                                                  # because torch.cuda
#                                                                  # .current_stream()
#                                                                  # .cuda_stream breaks
#                                                                  # the trace.
#
# We bridge the graph break with the ``quant_provenance`` registry:
#
#  * Producer (outer subgraph): rewrite ``contig = strided_view.contiguous()`` whose
#    result only leaves through the output node — call ``strided_producer_token(contig,
#    strided_view)`` that stores ``contig -> strided_view`` in the registry and returns
#    contig unchanged.
#  * Consumer (rmsnorm mini-subgraph): rewrite the rmsnorm whose input is a placeholder
#    — call ``strided_rmsnorm_consumer(output, contig, weight, eps, stream_id)`` that
#    looks up ``contig`` in the registry. If a strided source is found, call
#    ``fused_strided_rmsnorm(strided, weight, eps)`` and copy into output. Otherwise fall
#    back to the original rmsnorm.


def strided_producer_token(
    contig: torch.Tensor, strided_source: torch.Tensor
) -> torch.Tensor:
    """Producer-token: remember strided_source under contig key and return contig.

    Runs at the very end of the outer subgraph; the rmsnorm mini-subgraph that
    consumes ``contig`` then looks up the strided source via the registry.
    """
    from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import (
        remember_strided,
    )

    remember_strided(contig, strided_source)
    return contig


def strided_rmsnorm_consumer(
    output: torch.Tensor,
    contig: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    stream_id: int,
) -> None:
    """Consumer-token: write ``rmsnorm(input)`` into ``output``.

    With the new producer-side rewrite, ``contig`` is the strided view
    itself (the producer deletes the ``.contiguous()`` node from the FX
    graph). Uses the fp8+bf16 dual-output variant of
    ``fused_strided_rmsnorm`` so that the normed fp8 result is pushed to
    the slot cache for the downstream FP8 linear (e.g. q_b_proj).

    Falls back to bf16-only ``fused_strided_rmsnorm`` or
    ``rtp_llm_ops.rmsnorm`` on any error.
    """
    from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import (
        remember_quant,
        slot_push_quant,
    )

    # Fast path: compute bf16 + fp8 + scale in one kernel launch.
    # Store (fp8, scale) keyed on output for CudaFp8GEMMLinear.forward's
    # lookup_quant check, and push to slot cache as backup.
    try:
        _, fp8, scale = fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
            contig,
            weight,
            float(eps),
            group_size=128,
            scale_ue8m0=True,
            out_bf16=output,
        )
        remember_quant(output, fp8, scale)
        slot_push_quant(fp8, scale, tag="strided")
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "GraphFX strided_rmsnorm fp8 fast path failed shape=%s stride=%s: %s",
            tuple(contig.shape),
            tuple(contig.stride()),
            exc,
        )

    # Fallback: bf16-only norm (no fp8 push).
    try:
        fused_strided_rmsnorm(contig, weight, float(eps), out=output)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("GraphFX strided_rmsnorm consumer fallback: %s", exc)

    # Baseline path — preserve original semantics.
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    original = getattr(rtp_llm_ops, "rmsnorm", None)
    graphfx_original = getattr(original, "_graphfx_original", None)
    callable_ = graphfx_original or original
    if callable_ is None:
        raise RuntimeError(
            "rtp_llm_ops.rmsnorm unavailable; strided consumer fallback failed"
        )
    callable_(output, contig, weight, float(eps), int(stream_id))


def _rewrite_producer_outer(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Eliminate ``contig = strided_view.contiguous()`` entirely when its only
    consumers are layout-only nodes or the graph output — in that case the
    downstream rmsnorm subgraph can consume the strided view directly.

    The previous implementation inserted a ``strided_producer_token`` AFTER
    the contiguous node and relied on the consumer doing a runtime
    ``lookup_strided(contig)`` to find the original strided source. That left
    the original ``.contiguous()`` call in the graph, so it still ran at
    runtime and emitted a ``direct_copy_kernel`` for every layer/request.

    Now: redirect users of ``contig`` to ``strided`` directly and DCE the
    contiguous node. The downstream rmsnorm subgraph's placeholder is bound
    to the strided view at runtime; the consumer rewrite calls
    ``fused_strided_rmsnorm`` which handles strided input natively.
    """
    if not is_call_method(node, "contiguous"):
        return False
    if not node.args:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    if not _is_from_split(src):
        return False
    # Only rewrite when contig is consumed only via graph output (cross-graph case).
    if not all_users_are_layout_or_output_only(node, skip=set()):
        return False
    # Replace all users of the contig node with the strided source itself,
    # then delete the contig node.
    for user in list(node.users):
        user.replace_input_with(node, src)
    if not node.users:
        gm.graph.erase_node(node)
    return True


def _rewrite_consumer_inner(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite ``rmsnorm_mutating(output, contig_ph, weight_ph, eps_ph, stream_ph)``
    in a placeholder-only mini-subgraph with a consumer token that uses the
    registry to find the strided source."""
    if not _is_rmsnorm_node(node):
        return False
    args = list(node.args)
    if len(args) < 4:
        return False
    output_arg, input_arg, weight_arg, eps_arg = args[0], args[1], args[2], args[3]
    stream_arg = args[4] if len(args) > 4 else None
    # Cross-graph case: the input is a placeholder (or a contiguous of a placeholder),
    # OR a direct split view (when producer rewrite eliminated .contiguous() in same graph).
    if not isinstance(input_arg, torch.fx.Node):
        return False
    if input_arg.op != "placeholder" and not _is_from_split(input_arg):
        return False
    # Avoid firing twice on the same node (idempotency).
    if any(
        user.op == "call_function" and user.target is strided_rmsnorm_consumer
        for user in node.users
    ):
        return False
    new_args: list[object] = [output_arg, input_arg, weight_arg, eps_arg]
    if stream_arg is not None:
        new_args.append(stream_arg)
    else:
        new_args.append(0)
    with gm.graph.inserting_before(node):
        consumer_call = gm.graph.call_function(
            strided_rmsnorm_consumer, args=tuple(new_args)
        )
    node.replace_all_uses_with(consumer_call)
    gm.graph.erase_node(node)
    return True


def apply_strided_rmsnorm_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    same_graph = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite_bf16(gm, node):
            same_graph += 1
    if same_graph:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    producer = 0
    for node in list(gm.graph.nodes):
        if _rewrite_producer_outer(gm, node):
            producer += 1
    if producer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    consumer = 0
    for node in list(gm.graph.nodes):
        if _rewrite_consumer_inner(gm, node):
            consumer += 1
    if consumer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    total = same_graph + producer + consumer
    if total:
        record_graphfx_fusion_hit(_PASS_NAME, total)
        logger.info(
            "GraphFX strided_rmsnorm pass: same_graph=%d producer=%d consumer=%d",
            same_graph,
            producer,
            consumer,
        )
    return gm


def register_strided_rmsnorm_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        _PASS_NAME,
        apply_strided_rmsnorm_fp8_quant_fx_pass,
        priority=15,
        env_gate="GRAPHFX_FUSED_STRIDED_RMSNORM_FP8_QUANT",
    )


register_strided_rmsnorm_fp8_quant_pass()
