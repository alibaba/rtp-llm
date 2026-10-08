"""Checkpoint-ordered V4.1 routing and complete shared/routed FFN execution."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_collective import V41CollectiveMoE
from rtp_llm.models_py.modules.dsv4.v41_expert import V41SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE, _assert_values


class V41Router(nn.Module):
    """FP32 sqrt-softplus router; correction bias affects selection only.

    The checkpoint BF16 matrix is promoted once on the final device. TF32 must
    be disabled. Invalid live values fail instead of silently changing routes;
    dead rows return zero weights and -1 IDs without consuming poisoned inputs.
    """

    def __init__(
        self,
        weight,
        bias,
        *,
        topk=6,
        gate_temp=1.0,
        route_scale=1.5,
        norm_topk_prob=True,
        max_rows=128
    ):
        super().__init__()
        if (
            weight.ndim != 2
            or min(weight.shape) <= 0
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or not weight.is_contiguous()
            or weight.requires_grad
            or weight.device.type not in ("cpu", "cuda")
            or bias.shape != (weight.shape[0],)
            or bias.dtype != torch.float32
            or bias.device != weight.device
            or not bias.is_contiguous()
            or bias.requires_grad
        ):
            raise ValueError(
                "V4.1 router requires BF16/FP32 [experts,dim] and FP32 bias"
            )
        if (
            type(topk) is not int
            or not 1 <= topk <= weight.shape[0]
            or type(max_rows) is not int
            or max_rows <= 0
            or type(norm_topk_prob) is not bool
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in (gate_temp, route_scale)
            )
        ):
            raise ValueError(
                "V4.1 router requires positive finite scales and valid bounds"
            )
        if weight.is_cuda:
            if torch.cuda.get_device_capability(weight.device) != (12, 0):
                raise RuntimeError("V4.1 router requires exact SM120")
            with torch.cuda.device(weight.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("V4.1 router preparation must precede capture")
        self.weight, self.bias = weight.float(), bias
        self.experts, self.dim = weight.shape
        self.topk, self.max_rows = topk, max_rows
        self.gate_temp, self.route_scale, self.norm_topk_prob = (
            gate_temp,
            route_scale,
            norm_topk_prob,
        )
        _assert_values(
            torch.isfinite(self.weight).all() & torch.isfinite(bias).all(),
            "V4.1 router weights must be finite",
        )

    def forward(self, hidden, valid_rows):
        rows = hidden.shape[0] if hidden.ndim == 2 else -1
        if (
            not 0 <= rows <= self.max_rows
            or hidden.shape != (rows, self.dim)
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.weight.device
            or not hidden.is_contiguous()
            or hidden.requires_grad
            or valid_rows.shape != (rows,)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != hidden.device
            or not valid_rows.is_contiguous()
        ):
            raise ValueError(
                "V4.1 router requires bounded contiguous BF16 rows and a device mask"
            )
        if hidden.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("V4.1 FP32 router requires TF32 disabled")
        clean = torch.where(valid_rows[:, None], hidden, 0).float()
        _assert_values(
            torch.isfinite(clean).all(), "V4.1 live router inputs must be finite"
        )
        scores = F.softplus(F.linear(clean, self.weight) / self.gate_temp).sqrt()
        ranking = scores + self.bias
        _assert_values(
            torch.isfinite(ranking).all(), "V4.1 router scores must be finite"
        )
        ids = ranking.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, ids)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        _assert_values(
            torch.isfinite(weights).all(), "V4.1 routing weights must be finite"
        )
        return (
            torch.where(valid_rows[:, None], weights, 0),
            torch.where(valid_rows[:, None], ids, -1).int(),
        )


class V41FeedForward(nn.Module):
    """Router -> routed EP/local sum -> add one shared expert -> BF16.

    Local execution is allowed only with the complete expert set. A collective
    executor must bind the same global layer and geometry; its explicit admit()
    remains a worker-startup collective outside capture. Empty ranks still call
    it. The caller owns the synchronized invocation/chunk schedule.
    """

    def __init__(self, layer_id, router, shared, routed):
        super().__init__()
        if type(layer_id) is not int or not 0 <= layer_id < 40:
            raise ValueError("V4.1 FFN requires a backbone global layer ID")
        if not isinstance(router, V41Router) or not isinstance(shared, V41SharedExpert):
            raise ValueError("V4.1 FFN requires its router and shared expert")
        if isinstance(routed, V41CollectiveMoE):
            plan, local = routed.plan, routed.local
            if (
                plan.layer_id != layer_id
                or plan.dispatch.hidden != router.dim
                or plan.dispatch.experts != router.experts
                or plan.dispatch.topk != router.topk
                or router.max_rows > plan.rows_per_rank
                or plan.intermediate_dim != shared.inter_dim
                or plan.swiglu_limit != shared.swiglu_limit
            ):
                raise ValueError(
                    "V4.1 FFN router/shared weights disagree with the EP plan"
                )
        elif isinstance(routed, V41LocalMoE):
            local = routed
            if local.expert_start_id != 0 or local.experts.experts != router.experts:
                raise ValueError("V4.1 local FFN must own the complete expert set")
        else:
            raise ValueError(
                "V4.1 FFN requires explicit local or collective routed experts"
            )
        if (
            router.dim != shared.dim
            or router.dim != local.experts.dim
            or shared.inter_dim != local.experts.inter_dim
            or shared.swiglu_limit != local.experts.swiglu_limit
            or router.weight.device != shared.w13.weight.device
            or router.weight.device != local.experts.device
            or (shared.backend == "eager") != (local.experts.backend == "eager")
            or router.max_rows > local.max_tokens
            or router.topk > local.max_topk
        ):
            raise ValueError("V4.1 FFN component geometry/device/backend mismatch")
        self.layer_id, self.router, self.shared, self.routed = (
            layer_id,
            router,
            shared,
            routed,
        )
        self.dim, self.device = router.dim, router.weight.device

    def forward(self, hidden, valid_rows):
        weights, ids = self.router(hidden, valid_rows)
        if isinstance(self.routed, V41CollectiveMoE):
            routed = self.routed(hidden, weights, ids, valid_rows=valid_rows)
        else:
            layout = (
                "masked"
                if hidden.shape[0] <= self.routed.masked_token_limit
                else "contiguous"
            )
            routed = self.routed(
                hidden, weights, ids, valid_rows=valid_rows, layout=layout
            )
        shared = self.shared(hidden, valid_rows)
        # Routed contributions already contain routing weights. Round only once
        # after adding the shared BF16 result to the complete FP32 routed sum.
        return (routed + shared.float()).bfloat16()


def build_v41_feed_forward(
    layer_id,
    layer_weights,
    routed,
    *,
    max_rows=128,
    topk=6,
    gate_temp=1.0,
    route_scale=1.5,
    norm_topk_prob=True,
    swiglu_limit=10.0,
    backend="flashinfer"
):
    """Bind router/shared loader names while retaining the dictionary on failure."""
    from rtp_llm.utils.model_weight import W

    router = V41Router(
        layer_weights[W.v4_router_w],
        layer_weights[W.v4_router_bias],
        topk=topk,
        max_rows=max_rows,
        gate_temp=gate_temp,
        route_scale=route_scale,
        norm_topk_prob=norm_topk_prob,
    )
    shared = V41SharedExpert(
        {
            "w13_w": layer_weights[W.v4_shared_w13_w],
            "w13_s": layer_weights[W.v4_shared_w13_s],
            "w2_w": layer_weights[W.v4_shared_w2_w],
            "w2_s": layer_weights[W.v4_shared_w2_s],
        },
        swiglu_limit=swiglu_limit,
        backend=backend,
    )
    return V41FeedForward(layer_id, router, shared, routed)
