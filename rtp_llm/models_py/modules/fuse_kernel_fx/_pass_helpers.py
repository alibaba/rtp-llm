"""Shared FX-walking primitives used by the GraphFX passes."""

from __future__ import annotations

import operator
from typing import Iterable, Optional

import torch


def target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def is_call_function(node: object, name: str) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and target_name(node.target) == name
    )


def is_call_method(node: object, name: str) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_method"
        and target_name(node.target) == name
    )


def is_layout_only_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_method" and target_name(node.target) in (
        "contiguous",
        "reshape",
        "view",
    ):
        return True
    return node.op == "call_function" and target_name(node.target) in (
        "reshape",
        "view",
    )


def unwrap_layout_only(node: object) -> object:
    while isinstance(node, torch.fx.Node) and is_layout_only_node(node):
        node = node.args[0] if node.args else node
    return node


def static_shape(node: object):
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_meta = (node.meta or {}).get("tensor_meta")
    return getattr(tensor_meta, "shape", None)


def static_last_dim(node: object) -> Optional[int]:
    shape = static_shape(node)
    if not shape:
        return None
    value = shape[-1]
    return value if isinstance(value, int) else None


def static_numel_1d(node: object) -> Optional[int]:
    shape = static_shape(node)
    if not shape or len(shape) != 1:
        return None
    value = shape[0]
    return value if isinstance(value, int) else None


def is_quant_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name = target_name(node.target)
    return (
        name == "sgl_per_token_group_quant_fp8"
        or "sgl_per_token_group_quant_fp8" in str(node.target)
    )


def quant_contract_ok(node: torch.fx.Node, *, group_size: int = 128) -> bool:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    # Custom_op emits positional: (x, group_size, eps, col_major, tma, ue8m0, silu_mul, masked_m)
    actual_group = kwargs.get("group_size", args[1] if len(args) > 1 else None)
    column_major = kwargs.get(
        "column_major_scales", args[3] if len(args) > 3 else False
    )
    scale_tma = kwargs.get("scale_tma_aligned", args[4] if len(args) > 4 else False)
    fuse_silu = kwargs.get("fuse_silu_and_mul", args[6] if len(args) > 6 else False)
    masked_m = kwargs.get("masked_m", args[7] if len(args) > 7 else None)
    return (
        actual_group == group_size
        and bool(column_major)
        and bool(scale_tma)
        and not bool(fuse_silu)
        and masked_m is None
    )


def quant_scale_ue8m0(node: torch.fx.Node, *, default: bool = False) -> bool:
    """Extract ``scale_ue8m0`` from a ``sgl_per_token_group_quant_fp8`` call node.

    Honours both kwarg form (Python call) and positional form (custom_op
    dispatch flattens kwargs into positional at index 5).
    """
    args = list(node.args)
    kwargs = dict(node.kwargs)
    return bool(kwargs.get("scale_ue8m0", args[5] if len(args) > 5 else default))


def quant_group_size(node: torch.fx.Node, *, default: int = 128) -> int:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    value = kwargs.get("group_size", args[1] if len(args) > 1 else default)
    return int(value)


def is_getitem(node: object, source: torch.fx.Node, index: int) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and node.args[0] is source
        and node.args[1] == index
    )


def replace_quant_uses(
    quant_node: torch.fx.Node, q_node: torch.fx.Node, s_node: torch.fx.Node
) -> None:
    for user in list(quant_node.users):
        if (
            user.op == "call_function"
            and user.target == operator.getitem
            and len(user.args) >= 2
            and user.args[0] is quant_node
        ):
            if user.args[1] == 0:
                user.replace_all_uses_with(q_node)
                continue
            if user.args[1] == 1:
                user.replace_all_uses_with(s_node)
                continue
        raise RuntimeError(
            f"unexpected quant tuple user target={target_name(user.target)}"
        )


def first_quant_consumer_of(
    value_node: torch.fx.Node,
) -> Optional[torch.fx.Node]:
    """Return the first ``sgl_per_token_group_quant_fp8`` consumer of ``value_node``.

    Walks through layout-only views. Returns None if no quant consumer exists.
    """
    visited: set[int] = set()

    def _visit(node: torch.fx.Node) -> Optional[torch.fx.Node]:
        if id(node) in visited:
            return None
        visited.add(id(node))
        for user in node.users:
            if is_quant_node(user):
                if user.args and unwrap_layout_only(user.args[0]) is value_node:
                    return user
                continue
            if is_layout_only_node(user):
                hit = _visit(user)
                if hit is not None:
                    return hit
        return None

    return _visit(value_node)


def has_non_skip_user(
    node: torch.fx.Node,
    skip: Iterable[torch.fx.Node],
    quant_node: Optional[torch.fx.Node] = None,
) -> bool:
    """Return True iff ``node`` has any user not in ``skip`` and not a layout-only quant proxy."""
    skip_set = set(skip)
    for user in node.users:
        if user in skip_set:
            continue
        if quant_node is not None and _is_quant_only_layout_user(user, quant_node):
            continue
        return True
    return False


def _is_quant_only_layout_user(user: torch.fx.Node, quant_node: torch.fx.Node) -> bool:
    if user is quant_node:
        return True
    if not is_layout_only_node(user):
        return False
    return bool(user.users) and all(
        _is_quant_only_layout_user(next_user, quant_node) for next_user in user.users
    )


def snapshot_users_excluding(
    node: torch.fx.Node, exclude: Iterable[torch.fx.Node]
) -> list[torch.fx.Node]:
    excluded = set(exclude)
    return [user for user in list(node.users) if user not in excluded]


_INERT_TENSOR_ATTRS = frozenset(
    {"is_meta", "shape", "dtype", "device", "ndim", "is_cuda", "is_contiguous"}
)


def is_inert_guard_node(node: torch.fx.Node) -> bool:
    """Dynamo-injected getattr guards (e.g. ``getattr(t, 'is_meta')``)."""
    if node.op != "call_function":
        return False
    if target_name(node.target) != "getattr":
        return False
    return len(node.args) >= 2 and node.args[1] in _INERT_TENSOR_ATTRS


def all_users_are_layout_or_output_only(
    node: torch.fx.Node,
    seen: set[torch.fx.Node] | None = None,
    skip: set[torch.fx.Node] | None = None,
) -> bool:
    """Whether every user of ``node`` is a layout view or graph output.

    Used by cross-graph producer detection: when a producer op's output only
    leaves the FX graph through the ``output`` node (possibly through
    layout-only views), the FP8 quant consumer must be in a later subgraph.
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
        return node.op == "placeholder"
    for user in relevant_users:
        if user.op == "output":
            continue
        if is_layout_only_node(user):
            if not all_users_are_layout_or_output_only(user, seen, skip):
                return False
            continue
        if is_inert_guard_node(user):
            continue
        return False
    return True
