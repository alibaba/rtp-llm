"""GraphFX replacement for the DSA indexer's ``_get_logits_head_gate`` chain.

Eager unfused chain (in Indexer._get_logits_head_gate after the DSV4-style
strip):

    x_f32 = x.contiguous().float()
    weight_f32 = weight.float() if weight.dtype != float32 else weight
    weights = (x_f32 @ weight_f32.T).unsqueeze(-1) * q_scale * scale_const

This pass detects that chain and rewrites it to:

    fused_logits_head_gate(x_bf16, q_scale, weight_bf16, scale_const,
                           fallback_proj=None)

which dispatches to the small-T / large-T Triton kernel for fused (cast +
GEMV + 2 elementwise muls).

Pattern matching is anchored on the terminal ``mul(*, scale_const)`` where
``scale_const`` is a Python float, then walks back through ``mul(_,
q_scale)`` → ``unsqueeze(_, -1)`` → ``matmul(x_f32, weight_T)`` →
``to(contiguous(x), float32)`` and the transposed weight cast.
"""

from __future__ import annotations

import logging
import operator
from typing import Optional

import torch

from rtp_llm.models_py.modules.fuse_kernel_fx._pass_helpers import target_name
from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    eliminate_dead_code_preserving_graphfx_side_effects,
    record_graphfx_fusion_hit,
    record_graphfx_fusion_miss,
    register_graphfx_fusion_pass,
)

try:
    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
        fused_logits_head_gate,
    )
except Exception:  # noqa: BLE001

    def fused_logits_head_gate(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("fused_logits_head_gate unavailable")


logger = logging.getLogger(__name__)

_PASS_NAME = "indexer_logits_head_gate_fx"


# ----- node-shape predicates ----------------------------------------------


def _is_call_function(node: object, name_substr: str) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    return name_substr in target_name(node.target) or name_substr in str(node.target)


def _is_mul(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and (
        node.target is operator.mul
        or "mul" in target_name(node.target)
        or "aten.mul" in str(node.target)
    ):
        return True
    if node.op == "call_method" and target_name(node.target) == "mul":
        return True
    return False


def _is_matmul(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function":
        s = str(node.target)
        n = target_name(node.target)
        if n in ("matmul", "mm", "bmm") or any(
            kw in s for kw in ("aten.mm", "aten.matmul", "aten.bmm")
        ):
            return True
    if node.op == "call_method" and target_name(node.target) in ("matmul", "mm"):
        return True
    return False


def _is_unsqueeze_neg1(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_method" and target_name(node.target) == "unsqueeze":
        return len(node.args) >= 2 and node.args[1] == -1
    if node.op == "call_function":
        s = str(node.target)
        if "unsqueeze" in s and len(node.args) >= 2 and node.args[1] == -1:
            return True
    return False


def _is_to_float32(node: object) -> Optional[torch.fx.Node]:
    """If node casts its input to float32, return the input node; else None."""
    if not isinstance(node, torch.fx.Node):
        return None
    # ``.float()`` method
    if node.op == "call_method" and target_name(node.target) == "float":
        return node.args[0] if node.args else None
    # ``aten._to_copy`` with dtype=torch.float32
    if node.op == "call_function":
        s = str(node.target)
        if "_to_copy" in s or target_name(node.target) == "to":
            kwargs = dict(node.kwargs)
            dtype = kwargs.get("dtype")
            if dtype == torch.float32:
                return node.args[0] if node.args else None
            # Some lowerings put dtype positionally — accept as best-effort.
            if len(node.args) >= 2 and node.args[1] == torch.float32:
                return node.args[0]
            # Otherwise assume it's a float cast (defensive).
            if dtype is None and len(node.args) == 1:
                return node.args[0]
    return None


def _is_contiguous(node: object) -> Optional[torch.fx.Node]:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_method" and target_name(node.target) == "contiguous":
        return node.args[0] if node.args else None
    if node.op == "call_function":
        s = str(node.target)
        if "contiguous" in s:
            return node.args[0] if node.args else None
    return None


def _is_transpose(node: object) -> Optional[torch.fx.Node]:
    """If node is weight.T or transpose(weight, -1, -2), return the underlying weight node."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_method":
        n = target_name(node.target)
        if n in ("t", "T"):
            return node.args[0] if node.args else None
        if n == "transpose":
            # transpose(self, dim0, dim1) — only accept the last-two swap
            if len(node.args) >= 3:
                a0, a1 = node.args[1], node.args[2]
                if {a0, a1} == {-1, -2} or {a0, a1} == {0, 1}:
                    return node.args[0]
    if node.op == "call_function":
        s = str(node.target)
        n = target_name(node.target)
        # Dynamo emits ``weight.T`` as ``call_function getattr(weight, "T")``.
        if n == "getattr" or s.endswith("getattr"):
            if len(node.args) >= 2 and node.args[1] in ("T", "t"):
                return node.args[0] if isinstance(node.args[0], torch.fx.Node) else None
        if "aten.t" in s or "aten.transpose" in s:
            if node.args:
                return node.args[0]
    return None


def _unwrap_x_to_bf16(x_f32_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """x_f32 = x.contiguous().float(). Return the bf16 x node, or None."""
    contig_node = _is_to_float32(x_f32_node)
    if contig_node is None:
        return None
    bf16 = _is_contiguous(contig_node)
    if bf16 is not None:
        return bf16
    # ``.float()`` might appear before ``.contiguous()`` depending on the
    # graph; accept either ordering.
    bf16 = contig_node
    return bf16 if isinstance(bf16, torch.fx.Node) else None


def _unwrap_weight_bf16(weight_T_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """weight_T = weight.float().T  OR weight.T.  Return the bf16 weight node."""
    pre = _is_transpose(weight_T_node)
    if pre is None:
        # Maybe weight_f32 was created via .T then .float()
        pre = _is_to_float32(weight_T_node)
        if pre is None:
            return None
        pre = _is_transpose(pre) or pre
    # ``pre`` may be the float-casted weight; unwrap one more level
    pre_unwrap = _is_to_float32(pre)
    if pre_unwrap is not None:
        return pre_unwrap
    return pre


def _scalar_value(arg: object) -> Optional[float]:
    if isinstance(arg, (float, int)) and not isinstance(arg, bool):
        return float(arg)
    return None


# ----- rewrite -------------------------------------------------------------


def _try_rewrite(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    # Anchor: terminal mul(_, scale_const).  The scale_const may be a Python
    # float, OR an FX node that itself is a small chain of ``.item()`` /
    # ``mul`` reductions — Dynamo lifts ``self.softmax_scale``-style attrs
    # to placeholders so ``softmax_scale * weights_scale`` shows up as
    # ``mul(item, item)`` in the graph.  Both forms are accepted.
    if not _is_mul(node):
        return False
    if len(node.args) != 2:
        return False
    scale_arg: object = node.args[1]
    scale_const = _scalar_value(scale_arg)
    if scale_const is None and not isinstance(scale_arg, torch.fx.Node):
        return False

    inner = node.args[0]
    if not _is_mul(inner):
        record_graphfx_fusion_miss(_PASS_NAME, "outer_mul_inner_not_mul")
        return False
    if len(inner.args) != 2:
        return False
    unsqueeze_node = inner.args[0]
    q_scale_node = inner.args[1]
    if not isinstance(q_scale_node, torch.fx.Node):
        record_graphfx_fusion_miss(_PASS_NAME, "q_scale_not_node")
        return False

    if not _is_unsqueeze_neg1(unsqueeze_node):
        record_graphfx_fusion_miss(_PASS_NAME, "no_unsqueeze_neg1")
        return False

    matmul_node = unsqueeze_node.args[0]
    if not _is_matmul(matmul_node):
        record_graphfx_fusion_miss(_PASS_NAME, "no_matmul")
        return False
    if not isinstance(matmul_node, torch.fx.Node) or len(matmul_node.args) < 2:
        return False

    x_f32_node = matmul_node.args[0]
    weight_T_node = matmul_node.args[1]
    if not isinstance(x_f32_node, torch.fx.Node) or not isinstance(
        weight_T_node, torch.fx.Node
    ):
        return False

    x_bf16 = _unwrap_x_to_bf16(x_f32_node)
    if x_bf16 is None:
        record_graphfx_fusion_miss(_PASS_NAME, "x_not_to_float32_of_contig")
        return False

    weight_bf16 = _unwrap_weight_bf16(weight_T_node)
    if weight_bf16 is None:
        record_graphfx_fusion_miss(_PASS_NAME, "weight_T_not_recognized")
        return False

    # Rewrite the terminal mul with the fused call. fallback_proj is None
    # at FX level — the fused kernel decides at runtime whether to take the
    # Triton fast path or its internal baseline. ``scale_const`` may be a
    # Python float OR the FX node carrying ``softmax_scale * weights_scale``.
    scale_arg_for_fused: object = scale_const if scale_const is not None else scale_arg
    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            fused_logits_head_gate,
            args=(x_bf16, q_scale_node, weight_bf16, scale_arg_for_fused),
            kwargs={"fallback_proj": None},
        )
    node.replace_all_uses_with(fused)
    return True


def apply_indexer_logits_head_gate_fx_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    replaced = 0
    for node in list(gm.graph.nodes):
        if _try_rewrite(gm, node):
            replaced += 1
    if replaced:
        eliminate_dead_code_preserving_graphfx_side_effects(gm)
        record_graphfx_fusion_hit(_PASS_NAME, replaced)
        logger.info("GraphFX indexer_logits_head_gate pass: %d", replaced)
    return gm


def register_indexer_logits_head_gate_pass() -> None:
    register_graphfx_fusion_pass(
        _PASS_NAME,
        apply_indexer_logits_head_gate_fx_pass,
        priority=12,
        env_gate="GRAPHFX_FUSED_INDEXER_LOGITS_HEAD_GATE",
    )


register_indexer_logits_head_gate_pass()
