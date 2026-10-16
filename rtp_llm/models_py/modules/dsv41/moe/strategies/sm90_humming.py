"""SM90 grouped MXFP4 experts, optionally behind the existing DeepEP protocol.

Gate/up and down each use one Humming grouped GEMM. Routing weights are
applied BEFORE down-input quantization, as in the private V4 Expert; neither
Humming nor DeepEP combine should apply them a second time.
"""

import logging
import os
from typing import Dict

import torch
from torch import nn

from .._sm90_humming_linear import SM90HummingGroupedLinear
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .deepep import DeepEPStrategy


def group_local_routes(indices: torch.Tensor, local_start: int, local_end: int):
    """Stable route-slot permutation and exclusive expert offsets (eager).

    A route slot identifies both the input token and its original top-k slot.
    Invalid / non-local slots are omitted, including DeepEP's -1 padding.
    Empty experts have repeated offsets and consume no GEMM rows.
    """
    flat = indices.reshape(-1)
    slots = torch.where((flat >= local_start) & (flat < local_end))[0]
    expert = flat.index_select(0, slots) - local_start
    order = torch.argsort(expert, stable=True)
    slots = slots.index_select(0, order)
    counts = torch.bincount(expert, minlength=local_end - local_start)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    return slots, offsets


class _GroupedLocalExperts(nn.Module):
    def __init__(self, cfg: MoeCfg):
        super().__init__()
        self.cfg = cfg

    def setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        self.native_a8 = os.environ.get("DSV41_SM90_NATIVE_A8", "0") == "1"
        linear_cls = SM90HummingGroupedLinear
        linear_kwargs = {}
        if self.native_a8:
            from .._sm90_humming_a8 import SM90HummingGroupedA8Linear

            linear_cls = SM90HummingGroupedA8Linear
            linear_kwargs["promotion_group_size"] = int(
                os.environ.get("DSV41_SM90_A8_PROMOTION_GROUP", "32")
            )

        w1 = layer_weights.pop(W.v4_routed_w1_w)
        s1 = layer_weights.pop(W.v4_routed_w1_s)
        w3 = layer_weights.pop(W.v4_routed_w3_w)
        s3 = layer_weights.pop(W.v4_routed_w3_s)
        # Concatenate bytes because torch.cat does not implement E8M0.
        w13 = torch.cat((w1, w3), dim=1)
        s13 = torch.cat((s1.view(torch.uint8), s3.view(torch.uint8)), dim=1).view(
            s1.dtype
        )
        del w1, s1, w3, s3
        self.w13 = linear_cls(w13, s13, **linear_kwargs)
        del w13, s13
        self.w2 = linear_cls(
            layer_weights.pop(W.v4_routed_w2_w),
            layer_weights.pop(W.v4_routed_w2_s),
            **linear_kwargs,
        )

    def _forward_into_buf(self, x, weights, indices, local_start, local_end):
        from ..expert import silu_mul_split
        from .._sm90_route_kernels import reduce_routes, routed_swiglu

        # The variable-length route permutation is intentionally eager. Do not
        # let a capture silently freeze one request's routing into a graph.
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM90 Humming MoE currently requires CUDA graphs disabled"
            )
        if (
            not getattr(self, "native_a8", False)
            and os.environ.get("DSV41_SM90_INDEXED_MOE", "0") == "1"
        ):
            from .._sm90_humming_indexed import try_forward_indexed

            indexed = try_forward_indexed(
                self, x, weights, indices, local_start, local_end
            )
            if indexed is not None:
                if x.shape[0] and not getattr(self, "_indexed_logged", False):
                    logging.info(
                        "SM90 indexed local MoE active: rows=%d topk=%d experts=%d",
                        x.shape[0], indices.shape[1], local_end - local_start,
                    )
                    self._indexed_logged = True
                return indexed
        slots, offsets = group_local_routes(indices, local_start, local_end)
        if slots.numel() == 0:
            return torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        topk = indices.shape[1]
        grouped_x = x.index_select(0, slots // topk).contiguous()
        gate_up = self.w13(grouped_x, offsets)
        if gate_up.is_cuda and gate_up.dtype in (torch.bfloat16, torch.float32):
            hidden = routed_swiglu(
                gate_up,
                weights,
                slots,
                clamp_limit=self.cfg.swiglu_limit,
                output_dtype=x.dtype,
            )
        else:
            gate, up = gate_up.float().chunk(2, dim=-1)
            hidden = silu_mul_split(
                gate.contiguous(), up.contiguous(), clamp_limit=self.cfg.swiglu_limit
            )
            hidden = hidden * weights.reshape(-1).index_select(
                0, slots
            ).float().unsqueeze(1)
        # These large grouped intermediates are dead before down GEMM. Let
        # the stream-aware allocator reuse them without touching any cache.
        del gate_up, grouped_x
        down = self.w2(hidden.to(x.dtype), offsets)
        del hidden
        if down.is_cuda and 1 <= topk <= 8 and down.shape[1] > 1:
            return reduce_routes(down, slots, x.shape[0], topk)
        # Restore route-slot order before reducing so no floating point atomic
        # accumulation or backend-specific top-k ordering affects the sum.
        slot_output = torch.zeros(
            (indices.numel(), x.shape[1]), dtype=torch.float32, device=x.device
        )
        slot_output.index_copy_(0, slots, down.float())
        return slot_output.view(x.shape[0], topk, x.shape[1]).sum(dim=1)


@register_strategy
class SM90HummingStrategy(RoutedExpertsStrategy):
    name = "sm90_humming"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 9
            and cfg.dim % 128 == 0
            and cfg.moe_inter_dim % 128 == 0
            and cfg.n_routed_experts % cfg.ep_size == 0
        )

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        self._local = _GroupedLocalExperts(cfg)
        self._decode_graph = os.environ.get("DSV41_SM90_DECODE_GRAPH", "0") == "1"
        self._decode_graph_max_rows = int(
            os.environ.get("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
        )
        if self._decode_graph and self._decode_graph_max_rows <= 0:
            raise ValueError("SM90 decode graph max rows must be positive")
        self._dispatch = None
        if cfg.ep_size > 1:
            # Reuse the existing top-k padding, local/global expert-id mapping,
            # empty-rank handling and BF16 combine protocol, replacing only
            # the compute object before any SM100 weight preparation occurs.
            self._dispatch = DeepEPStrategy(cfg)
            self._dispatch._local = self._local

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)
        logging.info(
            "V4.1 layer %d: SM90 Humming grouped MXFP4, activation group128 "
            "%s; FP8 promotion group=%d; EP=%d; eager routing; "
            "bounded indexed small-batch opt-in=%s",
            self.cfg.layer_id,
            "native FP8 WGMMA" if self._local.native_a8 else "E8M0 decoded for BF16 MMA",
            self._local.w13.promotion_group_size if self._local.native_a8 else 0,
            self.cfg.ep_size,
            not self._local.native_a8
            and os.environ.get("DSV41_SM90_INDEXED_MOE", "0") == "1",
        )

    def setup_runtime(self) -> None:
        if self._decode_graph and (
            not self._local.native_a8
            or self._local.w13.promotion_group_size != 32
            or self._local.w2.promotion_group_size != 32
        ):
            raise RuntimeError("SM90 decode graph requires native W4A8/K32")
        if self.cfg.ep_size > 1:
            from rtp_llm.models_py.distributed.deepep_wrapper import (
                DeepEPMode,
                DeepEPWrapper,
            )

            if DeepEPWrapper._instance is None:
                raise RuntimeError(
                    "SM90 Humming EP requires initialized DeepEP normal dispatch "
                    "(enable use_deepep_moe)"
                )
            if DeepEPWrapper._instance.mode != DeepEPMode.NORMAL:
                raise RuntimeError(
                    "SM90 Humming EP requires DeepEP normal mode "
                    "(disable use_deepep_low_latency)"
                )
            if self._decode_graph and DeepEPWrapper._instance.config.use_deepep_internode:
                raise RuntimeError("SM90 decode graph requires intranode DeepEP")
        if self._decode_graph:
            logging.info(
                "SM90 bounded native W4A8/K32 graph routing enabled: max_rows=%d EP=%d; "
                "normal BF16 dispatch/combine, fixed-capacity GPU routing",
                self._decode_graph_max_rows, self.cfg.ep_size,
            )

    def forward(self, x, weights, indices):
        if self._decode_graph and x.shape[0] <= self._decode_graph_max_rows:
            from .._sm90_humming_graph import forward_local_static, forward_normal_static

            if self._dispatch is not None:
                return forward_normal_static(
                    self, x, weights, indices, self._decode_graph_max_rows
                )
            return forward_local_static(
                self._local, x, weights, indices,
                self.cfg.local_expert_start, self.cfg.local_expert_end,
            )
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM90 Humming capture requires DSV41_SM90_DECODE_GRAPH=1 and "
                "input rows within DSV41_SM90_DECODE_GRAPH_MAX_ROWS"
            )
        if self._dispatch is not None:
            return self._dispatch(x, weights, indices)
        return self._local._forward_into_buf(
            x, weights, indices, self.cfg.local_expert_start, self.cfg.local_expert_end
        )
