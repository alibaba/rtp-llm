"""Planned fixed-shape SM120 EP transport and shared-weight eager fallback."""

from typing import cast

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK, FP8_BLOCK
from rtp_llm.models_py.kernels.cuda.sm120.grouped_decode.op import grouped_op
from rtp_llm.models_py.modules.factory.fused_moe.defs.decode_plan import DecodeMoePlan
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.ep_combine import (
    fp32_peer_sum,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.graph_decode import (
    Sm120DecodeMoeGeometry,
    decode_payload_offsets,
    local_decode_geometry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.workspace import (
    workspace_for,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_scatter_v2,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_masked_fp8_quant_packed,
)
from rtp_llm.models_py.triton_kernels.moe.sm120_ep_kernels import ep_gather
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.models_py.utils.math import ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase
from .sm120_decode import Sm120DecodeExecutor, Sm120DecodeExperts, validate_decode_group


def require_sm120_decode_graph_providers() -> None:
    """Check the actual linked packages before entering graph warmup."""
    try:
        from flashinfer import mxfp8_quantize

        from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v2

        grouped_op("cuda", masked=True)
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "SM120 decode graphs require the shared CUDA13 build with the "
            "private masked grouped FP8×FP4 GEMM, FlashInfer MXFP8 and "
            "RTP block-128 quantization v2"
        ) from error

    if not (callable(mxfp8_quantize) and callable(per_token_group_quant_fp8_v2)):
        raise RuntimeError(
            "SM120 decode graphs require callable FlashInfer MXFP8 and "
            "RTP block-128 quantization v2 providers"
        )


def _fixed_payload_views(payload, hidden, topk):
    scale_end, weight_end, _ = decode_payload_offsets(hidden, topk)
    return (
        payload[:, :hidden].view(torch.float8_e4m3fn),
        payload[:, hidden:scale_end],
        payload[:, scale_end:weight_end].view(torch.float32),
        payload[:, weight_end:].view(torch.int32),
    )


def _coalesce_routes(weights, ids, total_experts):
    """Combine duplicate expert assignments before bounded expert scatter."""
    valid = (ids >= 0) & (ids < total_experts) & (weights != 0)
    slot = torch.arange(ids.size(1), device=ids.device)
    same_expert = (ids.unsqueeze(-1) == ids.unsqueeze(-2)) & valid.unsqueeze(-2)
    has_earlier_match = (same_expert & (slot[:, None] > slot)).any(dim=-1)
    first = valid & ~has_earlier_match
    # All matching weights contribute exactly once. This preserves hash routing
    # even for checkpoint tables with repeated IDs in a token's route.
    combined = torch.where(same_expert, weights.unsqueeze(-2), 0.0).sum(dim=-1)
    return torch.where(first, combined, 0.0), torch.where(first, ids, -1)


def pack_fixed_payload(x, weights, ids, valid_rows, payload, total_experts):
    from flashinfer import mxfp8_quantize

    valid = valid_rows[:, None]
    x_q, scale = mxfp8_quantize(
        torch.where(valid, x, 0.0).contiguous(), is_sf_swizzled_layout=False
    )
    weights = torch.where(valid, weights.float(), 0.0)
    weights, ids = _coalesce_routes(weights, ids, total_experts)
    activations, scales, routes, experts = _fixed_payload_views(
        payload, x.size(1), weights.size(1)
    )
    activations.view(torch.uint8).copy_(torch.where(valid, x_q.view(torch.uint8), 0))
    # UE8M0 encoding for scale 1 avoids any provider-specific padding bytes.
    scales.copy_(torch.where(valid, scale.view(torch.uint8).reshape_as(scales), 127))
    routes.copy_(weights)
    experts.copy_(ids)


class Sm120GraphDecodeExperts(Sm120DecodeExperts):
    def _setup_weights(self, layer_weights) -> None:
        super()._setup_weights(layer_weights)
        self._masked_grouped_gemm = grouped_op(self._w13.device, masked=True)

    def _forward_coalesced(self, x, weights, ids, scale, g, start, ws):
        a_fp8, a_scale = self.quantize_grouped_input(x, scale)
        # Coalescing bounds these counts to g.rows <= g.expert_capacity.
        # Exclude zero-weight routes before both scatter and gather.
        ids = torch.where(weights != 0, ids, -1).contiguous()
        local_ids, masked_m = recompute_topk_ids_sum_expert_count(
            ids, current_expert_start_id=start, num_local_experts=g.local_experts
        )
        ep_scatter_v2(
            a_fp8,
            a_scale,
            local_ids,
            g.expert_capacity,
            ws.start_loc,
            ws.expert_x.view(-1, g.hidden),
            ws.expert_x_scale,
            ws.out_index,
            scale_ue8m0=True,
        )
        # DeepGEMM uses this only to select launch parameters. The capacity is
        # the worst case and each real M is loaded from masked_m on device.
        expected_m = min(g.expert_capacity, ceil_div(g.rows * g.topk, g.total_experts))
        self._masked_grouped_gemm(
            (ws.expert_x, ws.expert_x_scale),
            (self._w13, self._s13),
            ws.gate_up,
            masked_m,
            expected_m,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        # Match eager's clamp and BF16 rounding in the masked layout.
        silu_mul_masked_fp8_quant_packed(
            ws.gate_up,
            ws.down_in,
            ws.down_in_scale,
            masked_m,
            clamp_limit=self.cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        self._masked_grouped_gemm(
            (ws.down_in, ws.down_in_scale),
            (self._w2, self._s2),
            ws.down_out,
            masked_m,
            expected_m,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        ep_gather(
            ws.down_out.view(-1, g.hidden),
            local_ids,
            weights,
            ws.out_index,
            ws.partial,
        )


class Sm120GraphDecodeExecutor(Fp8Fp4ExecutorBase):
    """Dispatch by the host plan, using one set of local weights for both paths."""

    preserve_output_dtype = True
    execute_empty_inputs = True
    single_round_dispatch = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        Sm120DecodeExecutor.check_conditions(checker, config)
        role_matches = checker.check(getattr(config, "is_decode_role", False))
        tp_matches = checker.check(getattr(config, "attn_tp_size", 1) == 1)
        if role_matches and tp_matches:
            try:
                local_decode_geometry(config, 1)
                require_sm120_decode_graph_providers()
            except (ValueError, RuntimeError) as error:
                checker.check(False, sm120_graph_decode=str(error))

    def setup_weights(self, layer_weights) -> None:
        require_sm120_decode_graph_providers()
        # The eager executor owns the only registered copy of the experts.
        # Fixed execution uses the same module through its masked entry point.
        self._eager = Sm120DecodeExecutor(
            self.cfg,
            self.quant_config,
            layer_weights,
            expert_class=Sm120GraphDecodeExperts,
        )

    def _forward_routed(self, x, topk_weights, topk_ids, extra_expert_args):
        plan = (extra_expert_args or {}).get("decode_plan")
        return self.forward(x, topk_weights, topk_ids, plan=plan)

    def forward(
        self, x, weights, indices, *, plan: DecodeMoePlan | None = None
    ) -> torch.Tensor:
        if plan is None:
            raise RuntimeError("sm120_graph_decode requires a planned decode execution")
        plan.validate_rows(x)
        if not plan.fixed:
            valid = plan.valid_rows[:, None]
            return self._eager(
                x, torch.where(valid, weights, 0.0), torch.where(valid, indices, -1)
            )
        validate_decode_group(self.cfg)
        if not is_sm120(x.device):
            raise RuntimeError("SM120 decode requires an exact SM120 device")
        if x.ndim != 2 or x.size(1) != self.cfg.dim or x.dtype != torch.bfloat16:
            raise ValueError("SM120 fixed decode requires BF16 [rows, hidden] inputs")
        if weights.shape != indices.shape or weights.shape != (
            x.size(0),
            self.cfg.moe_k,
        ):
            raise ValueError(
                "SM120 fixed decode routing must match the planned rows and TopK"
            )
        g = local_decode_geometry(self.cfg, x.size(0))
        return self._forward_fixed(x, weights, indices, plan, g)

    def _forward_fixed(self, x, weights, indices, plan, g):
        ws = self._get_workspace(g, plan, x.device)
        group = torch.distributed.group.WORLD
        pack_fixed_payload(
            x, weights, indices, plan.valid_rows, ws.payload, g.total_experts
        )
        torch.distributed.all_gather_into_tensor(ws.gathered, ws.payload, group=group)
        all_x, all_scale, all_weights, all_indices = _fixed_payload_views(
            ws.gathered, g.hidden, g.topk
        )
        experts = cast(Sm120GraphDecodeExperts, self._eager.experts)
        experts._forward_coalesced(
            all_x,
            all_weights,
            all_indices,
            all_scale,
            g,
            self.cfg.local_expert_start,
            ws,
        )
        torch.distributed.all_to_all_single(ws.returned, ws.partial, group=group)
        return fp32_peer_sum(ws.returned, g.world, ws.output)

    def _get_workspace(self, g: Sm120DecodeMoeGeometry, plan, device):
        return workspace_for(self.cfg).decode_graph_workspace(plan, g, device)
