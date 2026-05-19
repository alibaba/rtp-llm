"""Minimal test utilities for the QWEN35 GraphFX fusion passes.

CPU-only; manipulates ``torch.fx.Graph`` nodes directly without relying on
the actual fused Triton kernels.  Phase 1 does not include a perf harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GraphTargets:
    targets: list[str]
    op_kinds: list[str]


def target_names(gm: torch.fx.GraphModule) -> list[str]:
    return [
        getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes
    ]


def graph_targets(gm: torch.fx.GraphModule) -> GraphTargets:
    targets = target_names(gm)
    op_kinds = [node.op for node in gm.graph.nodes]
    return GraphTargets(targets=targets, op_kinds=op_kinds)


def assert_targets(
    gm: torch.fx.GraphModule,
    *,
    required: Sequence[str] = (),
    forbidden: Sequence[str] = (),
) -> None:
    names = target_names(gm)
    missing = [name for name in required if name not in names]
    if missing:
        raise AssertionError(f"missing required FX targets {missing}; got {names}")
    present = [name for name in forbidden if name in names]
    if present:
        raise AssertionError(f"forbidden FX targets present {present}; got {names}")


def make_dummy_tensor_meta(shape: tuple[int, ...], dtype=torch.bfloat16):
    """Construct a TensorMetadata-like object usable by the pass shape gate."""

    class _Meta:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.stride = (1,)
            self.is_quantized = False
            self.qparams = {}
            self.requires_grad = False
            self.memory_format = None

    return _Meta(shape, dtype)
