"""Experimental direct adapter for FlashInfer's SM12x DeepSeek-FP8 MoE chain.

This deliberately uses FlashInfer's three native operations instead of its
``MoELayer`` convenience wrapper.  RTP owns routing and TP reduction, while
the upstream chain owns only local routed-expert execution:

  BF16 Q0 + route -> FP8 FC1/SwiGLU/Q1 -> FP8 FC2/fused BF16 scatter-add.

The adapter must remain local-only: it has no shared-expert or collective
semantics.  In particular, an output returned here is a TP partial when the
expert weights are TP-sharded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

_BACKEND_ENV = "MOE_TP_PREFILL_BACKEND"
_BACKEND_NAME = "flashinfer_sm12x"
_FLASHINFER_COMMIT = "14a98117aef1cfc116026cc4354870cab983bf86"
_BLOCK = 128

__all__ = ["FlashInferSm12xFp8Moe", "prepare_flashinfer_block_scales"]


def _backend_selected() -> bool:
    """Use the sole TP-prefill backend switch, with no adapter-only opt-in."""

    return os.environ.get(_BACKEND_ENV, "default").strip().lower() == _BACKEND_NAME


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"FlashInfer SM12x FP8: {message}")


def prepare_flashinfer_block_scales(
    scale: torch.Tensor,
    *,
    rows: int,
    cols: int,
    experts: int,
    name: str,
) -> torch.Tensor:
    """Convert RTP block scales to FlashInfer's FP32/K-major scale layout.

    The input must be canonical raw FP32 scales ``[E, rows/128, cols/128]``.
    Packed DeepGEMM scales are a different ABI; this adapter does not unpack
    them. The loader retains canonical scales from the exact quantization that
    produced the weight values, before packing them for DeepGEMM.
    """

    _require(
        rows % _BLOCK == 0 and cols % _BLOCK == 0,
        "weight dimensions must be 128-aligned",
    )
    raw_shape = (experts, rows // _BLOCK, cols // _BLOCK)
    if scale.dtype is not torch.float32:
        raise TypeError(
            f"FlashInfer SM12x FP8: {name} must be canonical raw float32 scales; "
            f"got {scale.dtype}. Refuse to unpack DeepGEMM UE8M0 or "
            "requantize weights implicitly."
        )
    _require(
        tuple(scale.shape) == raw_shape,
        f"{name} FP32 shape {tuple(scale.shape)} != {raw_shape}",
    )
    # FlashInfer's DeepSeek-FP8 kernels index scales as [K_block, N_block].
    return scale.transpose(-1, -2).contiguous()


@dataclass(frozen=True)
class _NativeOps:
    route: Callable
    make_workspace: Callable
    fc1: Callable
    fc2: Callable
    out_sf_shape: Callable


@lru_cache(maxsize=1)
def _load_native_ops() -> _NativeOps:
    """Import only at execution time so unsupported installations stay usable."""

    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_fc1_act_q1 import (
            cute_dsl_sm12x_fc1_act_q1_fp8,
            out_sf_shape,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_fc2_finalize import (
            cute_dsl_sm12x_fc2_finalize_fp8,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_q0_route_triton import (
            fp8_q0_route_triton,
            make_fp8_q0_route_workspace,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"FlashInfer SM12x FP8 requires FlashInfer commit {_FLASHINFER_COMMIT}'s CUDA-13 "
            "CuTe-DSL modules; no dependency is installed or upgraded by RTP."
        ) from exc
    return _NativeOps(
        route=fp8_q0_route_triton,
        make_workspace=make_fp8_q0_route_workspace,
        fc1=cute_dsl_sm12x_fc1_act_q1_fp8,
        fc2=cute_dsl_sm12x_fc2_finalize_fp8,
        out_sf_shape=out_sf_shape,
    )


class FlashInferSm12xFp8Moe:
    """Direct local routed-expert adapter for FlashInfer SM120 DeepSeek FP8.

    ``w1`` is ``[E, 2I, H]`` in the upstream's ``[up | gate]`` order and
    ``w2`` is ``[E, H, I]``.  The caller's precomputed ``topk_ids`` must be in
    ``[0, E)``; RTP's trusted router provides that contract.  The instance
    contains immutable converted weights.  Workspaces and a
    default output are intentionally allocated per forward: callers may submit
    a returned output to an asynchronous NCCL all-reduce, so reusing it would
    corrupt an in-flight collective.  An explicit ``output_tensor`` is allowed
    for a caller that owns its lifetime; it is zeroed before the native fused
    scatter-add.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        _require(w1.dtype is torch.float8_e4m3fn, "w1 must be float8_e4m3fn")
        _require(w2.dtype is torch.float8_e4m3fn, "w2 must be float8_e4m3fn")
        _require(w1.ndim == 3 and w2.ndim == 3, "w1/w2 must be rank-3")
        experts, two_intermediate, hidden = w1.shape
        _require(experts > 0, "w1 must contain at least one expert")
        _require(two_intermediate % 2 == 0, "w1 second dimension must be 2*I")
        intermediate = two_intermediate // 2
        _require(
            tuple(w2.shape) == (experts, hidden, intermediate), "w2 must be [E,H,I]"
        )
        _require(
            hidden % _BLOCK == 0 and intermediate % _BLOCK == 0,
            "H and I must be 128-aligned",
        )
        _require(
            w1.device == w2.device and w1.device.type == "cuda",
            "weights must share a CUDA device",
        )
        _require(
            w1_scale.device == w1.device and w2_scale.device == w1.device,
            "weight scales must share the weight CUDA device",
        )
        self.w1 = w1.contiguous()
        self.w2 = w2.contiguous()
        self.w1_scale = prepare_flashinfer_block_scales(
            w1_scale, rows=two_intermediate, cols=hidden, experts=experts, name="w1"
        )
        self.w2_scale = prepare_flashinfer_block_scales(
            w2_scale, rows=hidden, cols=intermediate, experts=experts, name="w2"
        )
        self.num_experts = experts
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.device = w1.device

    @staticmethod
    def is_supported(
        hidden_states: torch.Tensor | None = None,
    ) -> bool:
        """Cheap capability gate; native imports and JIT remain lazy."""

        if not _backend_selected() or not torch.cuda.is_available():
            return False
        device = (
            hidden_states.device if hidden_states is not None else torch.device("cuda")
        )
        if device.type != "cuda":
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) in ((12, 0), (12, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require(
            hidden_states.device == self.device, "hidden_states is on another device"
        )
        _require(hidden_states.dtype is torch.bfloat16, "hidden_states must be BF16")
        _require(
            tuple(hidden_states.shape[1:]) == (self.hidden_size,),
            "hidden_states must be [T,H]",
        )
        _require(
            topk_ids.device == self.device and topk_weights.device == self.device,
            "routing tensors are on another device",
        )
        _require(
            topk_ids.dtype in (torch.int32, torch.int64),
            "topk_ids must be int32 or int64",
        )
        _require(topk_weights.dtype is torch.float32, "topk_weights must be FP32")
        _require(
            topk_ids.ndim == 2 and topk_ids.shape == topk_weights.shape,
            "routing tensors must share [T,k]",
        )
        _require(
            topk_ids.shape[0] == hidden_states.shape[0],
            "routing token count differs from hidden_states",
        )
        _require(
            hidden_states.shape[0] > 0 and topk_ids.shape[1] > 0,
            "empty routed input is unsupported",
        )
        # The native API is int32-only.  RTP's trusted router guarantees IDs
        # are in [0, E), so deliberately avoid min/max .item() checks here:
        # they would synchronize the current stream for every layer/chunk.
        native_topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        if output_tensor is None:
            # Unlike ``zeros_like``, this is contiguous even if the source view
            # is strided; FlashInfer's FC2 finalize output requires contiguity.
            output_tensor = torch.zeros(
                hidden_states.shape,
                dtype=torch.bfloat16,
                device=self.device,
            )
        else:
            _require(
                output_tensor.device == self.device,
                "output_tensor is on another device",
            )
            _require(
                output_tensor.dtype is torch.bfloat16
                and output_tensor.shape == hidden_states.shape,
                "output_tensor must be BF16 [T,H]",
            )
            _require(output_tensor.is_contiguous(), "output_tensor must be contiguous")
            output_tensor.zero_()

        ops = _load_native_ops()
        # Native route owns all temporary route metadata.  Do not cache it across
        # chunks: separate calls may overlap through a caller's NCCL stream.
        workspace = ops.make_workspace(hidden_states, native_topk_ids, self.num_experts)
        offsets, token_map, route_weights, a_q, a_scale = ops.route(
            hidden_states,
            native_topk_ids,
            topk_weights,
            self.num_experts,
            workspace=workspace,
        )
        total_pairs = hidden_states.shape[0] * native_topk_ids.shape[1]
        q1 = torch.empty(
            (total_pairs, self.intermediate_size),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        sf1 = torch.zeros(
            ops.out_sf_shape(total_pairs, self.intermediate_size, self.num_experts),
            dtype=torch.float32,
            device=self.device,
        )
        ops.fc1(a_q, a_scale, self.w1, self.w1_scale, offsets, out_q=q1, out_sf=sf1)
        ops.fc2(
            q1,
            sf1,
            self.w2,
            self.w2_scale,
            offsets,
            token_map,
            route_weights,
            hidden_states.shape[0],
            out=output_tensor,
        )
        return output_tensor
