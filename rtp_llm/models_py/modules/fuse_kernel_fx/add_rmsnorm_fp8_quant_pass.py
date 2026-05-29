"""GraphFX replacement for Qwen3.5 / GLM5 add+RMSNorm -> FP8 quant chains.

Scenario
========
Qwen3.5 and GLM5 decoder layers run an in-place residual add + RMSNorm via
``RMSResNorm.forward`` -> ``rtp_llm_ops.fused_add_rmsnorm`` (mutating) and
then quantize the result with ``sgl_per_token_group_quant_fp8`` before
feeding a DeepGEMM linear:

    hidden, residual ──► fused_add_rmsnorm(hidden, residual, weight, eps)
                        # mutates hidden, residual; returns None
                                            │ (hidden tensor reused)
                                            ▼
                          sgl_per_token_group_quant_fp8(hidden, group=128, …)
                                            │
                                            ▼
                                    (fp8, scale) ─► DeepGEMM

The eager code already provides the fused kernels
``fused_add_rmsnorm_fp8_quant`` (single output) and
``fused_add_rmsnorm_fp8_quant_with_bf16_output`` (dual output for sites where
the BF16 normed value is also consumed, e.g. shared-expert path).  This pass
replaces the unfused subgraph with the appropriate fused kernel and erases
the original mutating ``fused_add_rmsnorm`` node when no other consumer reads
the post-mutation BF16 value.

Constraints
===========
* Same-FX-graph only; cross-graph cases bail out with a recorded miss.
* ``group_size=128`` and ``column_major_scales=True / scale_tma_aligned=True``
  (DeepGEMM contract).  Other quant kwargs cause a contract miss.
* ``hidden_states`` must be 2-D with last-dim divisible by 128; the runtime
  CUDA wrapper validates the actual shape/stride/dtype.

Cross-graph (Phase 2) is intentionally left to ``add_rmsnorm_runtime``.
"""

from __future__ import annotations

import logging
import operator
from typing import Optional

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    eliminate_dead_code_preserving_graphfx_side_effects,
    record_graphfx_fusion_hit,
    record_graphfx_fusion_miss,
    register_graphfx_fusion_pass,
)

try:
    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        fused_add_rmsnorm_fp8_quant,
        fused_add_rmsnorm_fp8_quant_with_bf16_output,
    )
except Exception:  # noqa: BLE001 - keep pass importable on CPU/no-triton dev shells

    def fused_add_rmsnorm_fp8_quant(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_quant unavailable: triton/CUDA build required"
        )

    def fused_add_rmsnorm_fp8_quant_with_bf16_output(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_quant_with_bf16_output unavailable: triton/CUDA build required"
        )


logger = logging.getLogger(__name__)


# Hidden dims known to be supported by the fused Triton kernel
# (fused_add_rmsnorm_fp8_quant requires H % 128 == 0 and H <= 8192).
_SUPPORTED_HIDDEN_DIMS = {
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
}


def _target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def _is_quant_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = _target_name(node.target)
    return (
        name == "sgl_per_token_group_quant_fp8"
        or "sgl_per_token_group_quant_fp8" in str(node.target)
    )


def _is_mutating_add_rmsnorm_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function":
        return False
    target_name = _target_name(node.target).lower()
    return "fused_add_rmsnorm_mutating" in target_name or target_name.endswith(
        "fused_add_rmsnorm"
    )


def _quant_contract_ok(node: torch.fx.Node) -> bool:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    # Custom_op emits positional: (x, group_size, eps, col_major, tma, ue8m0, silu_mul, masked_m)
    # Original wrapper emits kwargs: (x, group_size=128, eps=..., ...)
    group_size = kwargs.get("group_size", args[1] if len(args) > 1 else None)
    column_major = kwargs.get(
        "column_major_scales", args[3] if len(args) > 3 else False
    )
    scale_tma = kwargs.get("scale_tma_aligned", args[4] if len(args) > 4 else False)
    fuse_silu = kwargs.get("fuse_silu_and_mul", args[6] if len(args) > 6 else False)
    masked_m = kwargs.get("masked_m", args[7] if len(args) > 7 else None)
    return (
        group_size == 128
        and bool(column_major)
        and bool(scale_tma)
        and not bool(fuse_silu)
        and masked_m is None
    )


def _add_rmsnorm_args(node: torch.fx.Node):
    """Return (hidden, residual, weight, eps) from a fused_add_rmsnorm call.

    Both the wrapped pybind ``rtp_llm_graphfx::fused_add_rmsnorm_mutating``
    and the original ``rtp_llm_ops.fused_add_rmsnorm`` callsite use the same
    positional contract: ``(hidden, residual, weight, eps, stream_id)``.
    """
    args = list(node.args)
    if len(args) < 4:
        return None
    return args[0], args[1], args[2], args[3]


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


def _is_quant_only_layout_user(user: torch.fx.Node, quant_node: torch.fx.Node) -> bool:
    if user is quant_node:
        return True
    if not _is_layout_only_node(user):
        return False
    return bool(user.users) and all(
        _is_quant_only_layout_user(next_user, quant_node) for next_user in user.users
    )


def _static_shape(node: object):
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_meta = (node.meta or {}).get("tensor_meta")
    return getattr(tensor_meta, "shape", None)


def _static_last_dim(node: object) -> Optional[int]:
    shape = _static_shape(node)
    if not shape:
        return None
    value = shape[-1]
    return value if isinstance(value, int) else None


def _static_numel_1d(node: object) -> Optional[int]:
    shape = _static_shape(node)
    if not shape or len(shape) != 1:
        return None
    value = shape[0]
    return value if isinstance(value, int) else None


def _hidden_contract_ok(hidden: object, weight: object) -> bool:
    """Reject patterns whose fixed hidden dim is not in the supported set.

    Dynamic batch / token dims are intentionally ignored; the fused Triton
    kernel validates them at runtime through assertion + shape checks.
    """
    k = _static_last_dim(hidden)
    if k is not None and k not in _SUPPORTED_HIDDEN_DIMS:
        return False
    weight_k = _static_numel_1d(weight)
    if weight_k is not None and weight_k not in _SUPPORTED_HIDDEN_DIMS:
        return False
    if k is not None and weight_k is not None and k != weight_k:
        return False
    return True


def _find_quant_consumer(
    hidden_node: torch.fx.Node,
) -> Optional[torch.fx.Node]:
    """Return the immediate ``sgl_per_token_group_quant_fp8`` consumer of hidden.

    Walks through layout-only views to find a quant node whose first input
    resolves back to ``hidden_node``.
    """
    candidates: list[torch.fx.Node] = []
    visited: set[int] = set()

    def _visit(node: torch.fx.Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for user in node.users:
            if _is_quant_node(user):
                if user.args and _unwrap_layout_only(user.args[0]) is hidden_node:
                    candidates.append(user)
                continue
            if _is_layout_only_node(user):
                _visit(user)

    _visit(hidden_node)
    if not candidates:
        return None
    # Use the first quant consumer (typically there's only one per layer)
    return candidates[0]


def _has_non_quant_consumer(
    hidden_node: torch.fx.Node,
    quant_node: torch.fx.Node,
    add_rmsnorm_node: torch.fx.Node,
) -> bool:
    """Whether hidden_node has consumers besides the quant chain or the mutating call."""
    for user in hidden_node.users:
        if user is add_rmsnorm_node:
            continue
        if user is quant_node:
            continue
        if _is_quant_only_layout_user(user, quant_node):
            continue
        return True
    return False


def _get_kw(node: torch.fx.Node, name: str, default):
    return node.kwargs.get(name, default)


def _replace_quant_uses(
    quant_node: torch.fx.Node, q_node: torch.fx.Node, s_node: torch.fx.Node
) -> None:
    for user in list(quant_node.users):
        if (
            user.op == "call_function"
            and user.target == operator.getitem
            and len(user.args) >= 2
        ):
            if user.args[0] is quant_node and user.args[1] == 0:
                user.replace_all_uses_with(q_node)
                continue
            if user.args[0] is quant_node and user.args[1] == 1:
                user.replace_all_uses_with(s_node)
                continue
        raise RuntimeError(
            "GraphFX add+RMSNorm+FP8 quant pass: unexpected quant tuple user "
            f"target={_target_name(user.target)}"
        )


def _snapshot_non_quant_consumers(
    hidden_node: torch.fx.Node,
    quant_node: torch.fx.Node,
    add_rmsnorm_node: torch.fx.Node,
) -> list[torch.fx.Node]:
    """Capture BF16 consumers of ``hidden_node`` BEFORE inserting fused nodes."""
    consumers: list[torch.fx.Node] = []
    for user in list(hidden_node.users):
        if user is add_rmsnorm_node:
            continue
        if user is quant_node:
            continue
        if _is_quant_only_layout_user(user, quant_node):
            continue
        consumers.append(user)
    return consumers


def _redirect_consumers(
    consumers: list[torch.fx.Node],
    hidden_node: torch.fx.Node,
    bf16_node: torch.fx.Node,
) -> None:
    for user in consumers:
        user.replace_input_with(hidden_node, bf16_node)


def _scale_ue8m0_from_quant(quant_node: torch.fx.Node) -> bool:
    # ``sgl_per_token_group_quant_fp8`` is wrapped as a torch.library.custom_op;
    # under dispatch the kwargs get flattened into positional args. scale_ue8m0
    # is the 6th arg (index 5). Without the positional fallback the pass would
    # silently emit fp32 scales when the downstream linear expects ue8m0/int32.
    from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import quant_scale_ue8m0

    return quant_scale_ue8m0(quant_node)


def _try_rewrite(gm: torch.fx.GraphModule, add_rmsnorm_node: torch.fx.Node) -> bool:
    if not _is_mutating_add_rmsnorm_node(add_rmsnorm_node):
        return False
    args = _add_rmsnorm_args(add_rmsnorm_node)
    if args is None:
        record_graphfx_fusion_miss(
            "add_rmsnorm_fp8_quant_fx", "fused_add_rmsnorm_args_unparsed"
        )
        return False
    hidden, residual, weight, eps = args
    if not isinstance(hidden, torch.fx.Node) or not isinstance(residual, torch.fx.Node):
        record_graphfx_fusion_miss("add_rmsnorm_fp8_quant_fx", "non_node_inputs")
        return False
    if not _hidden_contract_ok(hidden, weight):
        record_graphfx_fusion_miss(
            "add_rmsnorm_fp8_quant_fx", "unsupported_fixed_hidden_dim"
        )
        return False
    quant_node = _find_quant_consumer(hidden)
    if quant_node is None:
        record_graphfx_fusion_miss(
            "add_rmsnorm_fp8_quant_fx", "no_same_graph_quant_consumer"
        )
        return False
    if not _quant_contract_ok(quant_node):
        record_graphfx_fusion_miss(
            "add_rmsnorm_fp8_quant_fx", "quant_contract_mismatch"
        )
        return False

    has_bf16 = _has_non_quant_consumer(hidden, quant_node, add_rmsnorm_node)
    pre_existing_bf16_consumers = (
        _snapshot_non_quant_consumers(hidden, quant_node, add_rmsnorm_node)
        if has_bf16
        else []
    )
    scale_ue8m0 = _scale_ue8m0_from_quant(quant_node)
    quant_eps = _get_kw(quant_node, "eps", 1e-4)
    group_size = _get_kw(quant_node, "group_size", 128)

    # Always use the dual-output kernel. The original mutating op writes the
    # normed result back into hidden_states; Dynamo communicates this mutation
    # to later subgraphs via the tensor object itself (not through the graph
    # output). We must honour this contract by copying bf16_out back into
    # hidden_states even when no in-graph consumer reads the bf16 value.
    with gm.graph.inserting_before(add_rmsnorm_node):
        fused = gm.graph.call_function(
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
            args=(hidden, residual, weight),
            kwargs={
                "eps": eps,
                "group_size": group_size,
                "scale_ue8m0": scale_ue8m0,
            },
        )
        bf16_node = gm.graph.call_function(operator.getitem, args=(fused, 0))
        q_node = gm.graph.call_function(operator.getitem, args=(fused, 1))
        s_node = gm.graph.call_function(operator.getitem, args=(fused, 2))
        # Preserve the mutation contract: write the normed bf16 result back
        # into hidden_states so later subgraphs see the correct value.
        gm.graph.call_method("copy_", args=(hidden, bf16_node))

    _replace_quant_uses(quant_node, q_node, s_node)
    if has_bf16:
        _redirect_consumers(pre_existing_bf16_consumers, hidden, bf16_node)
    # Erase the original mutating add+RMSNorm only if no remaining users.
    if len(add_rmsnorm_node.users) == 0:
        gm.graph.erase_node(add_rmsnorm_node)
    return True


def _is_layout_only_or_quant_user(user: torch.fx.Node) -> bool:
    return _is_layout_only_node(user) or _is_quant_node(user)


_INERT_TENSOR_ATTRS = frozenset(
    {"is_meta", "shape", "dtype", "device", "ndim", "is_cuda", "is_contiguous"}
)


def _is_inert_guard_node(node: torch.fx.Node) -> bool:
    """Dynamo-injected getattr guards (e.g. ``getattr(t, 'is_meta')``)."""
    if node.op != "call_function":
        return False
    if _target_name(node.target) != "getattr":
        return False
    return len(node.args) >= 2 and node.args[1] in _INERT_TENSOR_ATTRS


def _all_users_are_layout_or_output_only(
    node: torch.fx.Node,
    seen: set[torch.fx.Node] | None = None,
    skip: set[torch.fx.Node] | None = None,
) -> bool:
    """Whether every user of ``node`` is a layout view or graph output.

    Used to decide that a given mutating ``fused_add_rmsnorm`` is purely a
    cross-graph producer (its bf16 result leaves the graph through the
    ``output`` node and there is no in-graph quant consumer).  Layout-only
    users like ``view`` / ``reshape`` are transparent passthroughs.

    ``skip`` lists nodes that should be ignored — typically the mutating
    ``fused_add_rmsnorm`` call itself, since the caller treats it as the
    producer of the post-mutation value rather than as an unrelated user.
    """
    if seen is None:
        seen = set()
    if skip is None:
        skip = set()
    if node in seen:
        return True
    seen.add(node)
    relevant_users = [user for user in node.users if user not in skip]
    if not relevant_users:
        # For mutating ops, the modified tensor flows to the next subgraph
        # via Dynamo's implicit mutation tracking — no explicit output node.
        # Return True so the producer token can register provenance.
        return node.op == "placeholder"
    for user in relevant_users:
        if user.op == "output":
            continue
        if _is_layout_only_node(user):
            if not _all_users_are_layout_or_output_only(user, seen, skip):
                return False
            continue
        if _is_inert_guard_node(user):
            continue
        return False
    return True


def _rewrite_cross_graph_producer(
    gm: torch.fx.GraphModule, add_rmsnorm_node: torch.fx.Node
) -> bool:
    """Replace ``fused_add_rmsnorm_mutating`` with the fused producer custom_op.

    The producer custom_op has ``mutates_args=("hidden_states", "residual")``,
    preserving Dynamo's mutation tracking across subgraph boundaries.
    Internally it runs ``fused_add_rmsnorm_fp8_quant_inplace`` (one Triton
    kernel, zero memcpy) and stashes ``(fp8, scale)`` in the provenance
    registry for the cross-graph consumer to look up.
    """
    args = _add_rmsnorm_args(add_rmsnorm_node)
    if args is None:
        return False
    hidden, residual, weight, eps = args
    if not isinstance(hidden, torch.fx.Node) or not isinstance(residual, torch.fx.Node):
        return False
    if not _hidden_contract_ok(hidden, weight):
        record_graphfx_fusion_miss(
            "add_rmsnorm_fp8_quant_fx", "producer_token_unsupported_fixed_hidden_dim"
        )
        return False
    if not _all_users_are_layout_or_output_only(hidden, skip={add_rmsnorm_node}):
        return False

    from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_runtime import (
        build_producer_custom_op,
    )

    producer_op = build_producer_custom_op()
    if producer_op is None:
        return False

    with gm.graph.inserting_before(add_rmsnorm_node):
        gm.graph.call_function(
            producer_op,
            args=(hidden, residual, weight),
            kwargs={
                "eps": eps,
                "group_size": 128,
                "scale_ue8m0": True,
            },
        )
    if len(add_rmsnorm_node.users) == 0:
        gm.graph.erase_node(add_rmsnorm_node)
    return True


def _rewrite_cross_graph_consumer(
    gm: torch.fx.GraphModule, quant_node: torch.fx.Node
) -> bool:
    """Replace an isolated quant call whose input is a graph placeholder.

    Fires when the producer ``fused_add_rmsnorm`` is in a different FX graph
    so the quant input has no matching producer here.  The consumer-token
    function checks the runtime provenance registry; when the registry has a
    precomputed (fp8, scale) it returns them directly, otherwise it falls
    back to ``sgl_per_token_group_quant_fp8``.
    """
    if not _is_quant_node(quant_node):
        return False
    if not _quant_contract_ok(quant_node):
        return False
    quant_arg = quant_node.args[0] if quant_node.args else None
    if quant_arg is None:
        return False
    x_node = _unwrap_layout_only(quant_arg)
    if not isinstance(x_node, torch.fx.Node):
        return False
    # If the input is a same-graph add_rmsnorm output, skip — the same-graph
    # rewrite (``_try_rewrite``) already handled it.
    if any(_is_mutating_add_rmsnorm_node(user) for user in (x_node, *x_node.users)):
        return False
    if x_node.op != "placeholder" and not _is_layout_only_node(x_node):
        # Conservative: only fire on placeholder / layout-only chains so we
        # don't accidentally redirect quants that have a same-graph producer
        # we just didn't recognise.
        return False

    from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
        quant_group_size as _qgs,
    )
    from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import (
        quant_scale_ue8m0 as _qsue8m0,
    )
    from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_runtime import (
        graphfx_fused_add_rmsnorm_fp8_quant_from_provenance,
    )

    fallback_y = quant_arg if quant_arg is not x_node else None
    with gm.graph.inserting_before(quant_node):
        fused = gm.graph.call_function(
            graphfx_fused_add_rmsnorm_fp8_quant_from_provenance,
            args=(x_node,),
            kwargs={
                "fallback_y": fallback_y,
                "group_size": _qgs(quant_node),
                "eps": quant_node.kwargs.get("eps", 1e-4),
                "column_major_scales": quant_node.kwargs.get(
                    "column_major_scales", True
                ),
                "scale_tma_aligned": quant_node.kwargs.get("scale_tma_aligned", True),
                "scale_ue8m0": _qsue8m0(quant_node),
                "fuse_silu_and_mul": quant_node.kwargs.get("fuse_silu_and_mul", False),
                "masked_m": quant_node.kwargs.get("masked_m", None),
            },
        )
    quant_node.replace_all_uses_with(fused)
    return True


def apply_add_rmsnorm_fp8_quant_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    same_graph = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            same_graph += 1
    if same_graph:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    # Cross-graph producer rewrite: replace the mutating ``fused_add_rmsnorm``
    # node with the producer custom_op that calls
    # ``fused_add_rmsnorm_fp8_quant_with_bf16_output`` and stashes
    # ``(fp8, scale)`` in the provenance registry for the cross-graph
    # consumer to pick up.
    producer = 0
    for node in list(gm.graph.nodes):
        if _is_mutating_add_rmsnorm_node(node) and _rewrite_cross_graph_producer(
            gm, node
        ):
            producer += 1
    if producer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    # Cross-graph consumer rewrite: generic — rewrites any standalone
    # ``sgl_per_token_group_quant_fp8`` node whose input is a placeholder
    # (i.e. came from a different subgraph) into a consumer-token call that
    # looks up the provenance registry. Provenance hits come from upstream
    # producer-side rewrites (``silu_and_mul``, ``strided_rmsnorm`` already
    # populate the registry). On miss the consumer-token falls back to a
    # standalone quant, preserving correctness.
    consumer = 0
    for node in list(gm.graph.nodes):
        if _rewrite_cross_graph_consumer(gm, node):
            consumer += 1
    if consumer:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)

    total = same_graph + producer + consumer
    if total:
        record_graphfx_fusion_hit("add_rmsnorm_fp8_quant_fx", total)
        logger.info(
            "GraphFX add+RMSNorm+FP8 quant pass: same_graph=%d producer=%d consumer=%d",
            same_graph,
            producer,
            consumer,
        )
    return gm


def register_add_rmsnorm_fp8_quant_pass() -> None:
    register_graphfx_fusion_pass(
        "add_rmsnorm_fp8_quant_fx",
        apply_add_rmsnorm_fp8_quant_fx_pass,
        priority=10,
        env_gate="GRAPHFX_FUSED_ADD_RMSNORM_FP8_QUANT",
    )


register_add_rmsnorm_fp8_quant_pass()
