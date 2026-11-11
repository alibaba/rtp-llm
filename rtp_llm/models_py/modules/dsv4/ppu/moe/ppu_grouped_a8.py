"""TP Prefill experts with official A8 activation and FP4 weight semantics."""

import os

import torch
from torch import nn

from rtp_llm.models_py.kernels.ppu.moe.ppu_a8_moe import (
    quantize_a8,
    repack_fp4_to_fp8,
    weighted_nopad_a8,
)
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_fp4 import (
    _derive_inter_local_and_tp,
)
from rtp_llm.utils.model_weight import W


class PpuGroupedA8Strategy(nn.Module):
    name = "ppu_grouped_a8"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._bn128_stage2 = (
            os.environ.get("DSV4_PREFILL_A8_BN128_STAGE2", "0") == "1"
            and cfg.tp_size == 8
            and cfg.ep_size == 1
        )

    def _grouped_launch_kwargs(self, lhs, rhs):
        if (
            self._bn128_stage2
            and tuple(lhs.shape) in ((98304, 4096), (98304, 2048))
            and tuple(rhs.shape) == (256, 512, lhs.shape[1])
        ):
            from deep_gemm.jit_kernels.gemm_fp8 import get_smem_config

            return {
                "configs": (
                    38,
                    192,
                    128,
                    128,
                    48,
                    64,
                    2,
                    get_smem_config(2, lhs.shape[1], 192, 128, 128),
                )
            }
        return {}

    def setup_weights(self, layer_weights):
        keys = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        )
        w1, s1, w2, s2, w3, s3 = (layer_weights[k] for k in keys)
        self.inter_local, self.routed_tp_size = _derive_inter_local_and_tp(
            self.cfg,
            tuple(w1.shape),
            tuple(w2.shape),
            tuple(w3.shape),
            tuple(s1.shape),
            tuple(s2.shape),
            tuple(s3.shape),
        )
        w13, s13 = repack_fp4_to_fp8(torch.cat((w1, w3), 1), torch.cat((s1, s3), 1))
        import torch.distributed as dist

        if dist.get_world_size() != self.cfg.tp_size:
            raise ValueError("A8 output sharding requires the TP WORLD group")
        self.output_dim = self.cfg.dim // self.cfg.tp_size
        self.output_start = dist.get_rank() * self.output_dim

        # Preserve the reference's full-K accumulation before BF16 rounding.
        def redistribute(tensor):
            send = torch.stack(
                list(tensor.view(torch.uint8).chunk(self.cfg.tp_size, dim=1))
            ).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send)
            return torch.cat(list(recv.unbind(0)), dim=2).contiguous()

        w2 = redistribute(w2)
        s2 = redistribute(s2).view(torch.float8_e8m0fnu)
        w2, s2 = repack_fp4_to_fp8(w2, s2)
        for name, tensor in zip(("_w13", "_s13", "_w2", "_s2"), (w13, s13, w2, s2)):
            self.register_buffer(name, tensor, persistent=False)
        for key in keys:
            layer_weights.pop(key)

    def forward(
        self,
        x,
        weights,
        indices,
        *,
        shared=None,
        route_scale=None,
        shared_schedule=None
    ):
        from deep_gemm.jit_kernels.m_grouped_gemm_fp8_k32 import (
            m_grouped_gemm_fp8_fp8_bf16_nt_nopad as grouped,
        )

        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
            gather_local_loop_compatible,
        )

        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or x.shape[1] != self.cfg.dim
            or weights.shape != indices.shape
            or indices.shape[0] != x.shape[0]
            or weights.dtype != torch.float32
            or indices.dtype != torch.int64
            or weights.device != x.device
            or indices.device != x.device
        ):
            raise ValueError("Invalid A8 Prefill tokens/routes")
        if (shared is None) != (route_scale is None) or route_scale not in (
            None,
            1.0,
            1,
        ):
            raise ValueError(
                "A8 routes must include route_scale before W2 quantization"
            )
        if x.shape[0] == 0:
            return torch.zeros_like(x, dtype=torch.float32)
        indices, weights = indices.contiguous(), weights.contiguous()
        source, experts, inverse, counts = _compact_routes_for_tp(
            indices, self.cfg.n_local_experts
        )
        q, scale = quantize_a8(x.contiguous())
        compact_q = q.index_select(0, source)
        compact_scale = scale.index_select(0, source).t().contiguous().t()
        total = indices.numel()
        gate_up = torch.empty(
            (total, 2 * self.inter_local), device=x.device, dtype=torch.bfloat16
        )
        if shared_schedule is not None:
            shared_schedule.before_up()
        grouped(
            (compact_q, compact_scale),
            (self._w13, self._s13),
            gate_up,
            experts,
            counts,
            **self._grouped_launch_kwargs(compact_q, self._w13),
        )
        if shared_schedule is not None:
            shared_schedule.after_up()
        hidden = weighted_nopad_a8(
            gate_up, source, experts, indices, weights, counts, self.cfg.swiglu_limit
        )
        hidden = (
            _all_gather_columns(hidden[0], self.cfg.tp_size),
            _all_gather_columns(hidden[1].contiguous(), self.cfg.tp_size)
            .t()
            .contiguous()
            .t(),
        )
        down = torch.empty(
            (total, self.output_dim), device=x.device, dtype=torch.bfloat16
        )
        if shared_schedule is not None:
            shared_schedule.before_down()
        grouped(
            hidden,
            (self._w2, self._s2),
            down,
            experts,
            counts,
            **self._grouped_launch_kwargs(hidden[0], self._w2),
        )
        # Router weights have already been applied inside weighted_nopad_a8.
        ones = torch.ones_like(weights)
        if shared is not None:
            shared = shared[
                :, self.output_start : self.output_start + self.output_dim
            ].contiguous()
            out = torch.empty_like(shared, dtype=torch.float32)
            gather_and_combine_local_loop_compatible(
                down, indices, ones, inverse, shared, out, 1.0, keep_fp32=True
            )
        else:
            out = torch.empty(
                (x.shape[0], self.output_dim), device=x.device, dtype=torch.float32
            )
            gather_local_loop_compatible(down, indices, ones, inverse, out, fused=True)
        return _all_gather_columns(
            out.bfloat16() if shared is not None else out, self.cfg.tp_size
        )


@torch.no_grad()
def _compact_routes_for_tp(topk_ids, num_experts):
    # Every TP rank must gather the same expert rows in the same order.
    flat = topk_ids.reshape(-1)
    valid = (flat >= 0) & (flat < num_experts)
    keys = torch.where(valid, flat, num_experts)
    order = torch.argsort(keys, stable=True)
    sorted_ids = flat.index_select(0, order)
    sorted_valid = valid.index_select(0, order)
    sources = torch.where(sorted_valid, order // topk_ids.shape[1], 0)
    experts = torch.where(sorted_valid, sorted_ids, 0).to(torch.int32)
    inverse = torch.empty_like(flat)
    inverse.scatter_(
        0,
        order,
        torch.where(sorted_valid, torch.arange(flat.numel(), device=flat.device), -1),
    )
    counts = torch.zeros(num_experts, device=flat.device, dtype=torch.int32)
    counts.scatter_add_(0, flat.clamp(0, num_experts - 1), valid.to(torch.int32))
    return sources, experts, inverse.view_as(topk_ids), counts


def _all_gather_columns(tensor, size):
    import torch.distributed as dist

    original_dtype = tensor.dtype
    if original_dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8)
    outputs = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(outputs, tensor)
    return torch.cat(outputs, dim=1).view(original_dtype)
