"""Instance-selected V4 routed experts on the engine's PPU DeepEP LL group."""

import torch


def take_raw_routed_mxfp4_weights(cfg, layer_weights):
    """Consume validated EP-local checkpoint nibbles and raw E8M0 scales."""
    from rtp_llm.utils.model_weight import W

    keys = (
        W.v4_routed_w1_w,
        W.v4_routed_w1_s,
        W.v4_routed_w2_w,
        W.v4_routed_w2_s,
        W.v4_routed_w3_w,
        W.v4_routed_w3_s,
    )
    w1, s1, w2, s2, w3, s3 = (layer_weights[k] for k in keys)
    e, d, inter = cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim
    expected = ((e, inter, d // 2), (e, d, inter // 2), (e, inter, d // 2))
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    for w, s, shape in zip((w1, w2, w3), (s1, s2, s3), expected):
        if (
            w.shape != shape
            or w.dtype not in (torch.uint8, torch.int8)
            or s.shape != (*shape[:-1], shape[-1] // 16)
            or e8m0 is None
            or s.dtype != e8m0
            or w.device != w1.device
            or s.device != w1.device
        ):
            raise ValueError(
                "PPU routed weights require EP-local MXFP4/E8M0 checkpoint geometry"
            )
    if not w1.is_cuda:
        raise RuntimeError("PPU routed MXFP4 requires PPU-resident weights")
    result = (
        torch.cat((w1.view(torch.uint8), w3.view(torch.uint8)), dim=1).contiguous(),
        torch.cat((s1, s3), dim=1).contiguous(),
        w2.view(torch.uint8).contiguous(),
        s2.contiguous(),
    )
    for key in keys:
        layer_weights.pop(key)
    return result


def prepare_routed_mxfp4_weights(cfg, layer_weights):
    """Preserve EP-sliced checkpoint nibbles; prepare scale pairs once."""
    from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import prepare_fp4_weight_scale_mxfp4

    w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(cfg, layer_weights)
    return (
        w13,
        prepare_fp4_weight_scale_mxfp4(s13),
        w2,
        prepare_fp4_weight_scale_mxfp4(s2),
    )


class PpuDeepEPFP4Strategy(torch.nn.Module):
    """DP/EP routed compute; selection is explicit, never process-global."""

    name = "ppu_deepep_fp4"

    @classmethod
    def can_handle(cls, cfg):
        return (
            cfg.tp_size > 0
            and cfg.ep_size > 0
            and cfg.n_local_experts > 0
            and 0 <= cfg.ep_rank < cfg.ep_size
            and cfg.local_expert_start == cfg.ep_rank * cfg.n_local_experts
            and cfg.local_expert_end == (cfg.ep_rank + 1) * cfg.n_local_experts
            and cfg.n_routed_experts == cfg.n_local_experts * cfg.ep_size
            and cfg.dim > 0
            and cfg.dim % 128 == 0
            and cfg.moe_inter_dim > 0
            and cfg.moe_inter_dim % 256 == 0
            and 0 < cfg.n_activated_experts <= min(16, cfg.n_routed_experts)
            and cfg.max_tokens_per_rank > 0
        )

    def __init__(
        self,
        cfg,
        *,
        expected_m_policy="capacity",
        output_dtype=torch.float32,
        gemm_tile="auto",
    ):
        super().__init__()
        self.cfg = cfg
        if output_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("PPU routed output must be FP32 or BF16")
        self.output_dtype = output_dtype
        if expected_m_policy not in ("capacity", "batch"):
            raise ValueError("PPU MoE expected rows policy must be capacity or batch")
        self._expected_m_policy = expected_m_policy
        if gemm_tile not in ("auto", "n128"):
            raise ValueError("PPU MoE GEMM tile must be auto or n128")
        self._gemm_tile = gemm_tile
        if not self.can_handle(cfg):
            raise ValueError("PPU DeepEP MXFP4 requires compatible EP-local experts")
        self._wrapper = None
        # At TP>1 the wrapper slices the replicated batch before dispatch, so
        # the per-rank MoE budget is this rank's slice of the global capacity.
        dispatch_budget = (cfg.max_tokens_per_rank + max(cfg.tp_size, 1) - 1) // max(
            cfg.tp_size, 1
        )
        self._expected_m = max(
            1,
            (
                dispatch_budget * cfg.ep_size * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )

    def expected_rows(self, num_tokens):
        """Host launch hint only; never truncate the buffer or device counts."""
        if self._expected_m_policy == "batch":
            # Remote ranks may send far more rows than the local estimate.
            # The masked kernels still traverse their full device counts;
            # this value changes launch geometry, never receive capacity.
            cfg = self.cfg
            return max(
                1,
                (
                    num_tokens * cfg.ep_size * cfg.n_activated_experts
                    + cfg.n_routed_experts
                    - 1
                )
                // cfg.n_routed_experts,
            )
        return self._expected_m

    def setup_weights(self, layer_weights):
        weights = prepare_routed_mxfp4_weights(self.cfg, layer_weights)
        for name, value in zip(("_w13", "_s13", "_w2", "_s2"), weights):
            self.register_buffer(name, value, persistent=False)

    def _bind_wrapper(self):
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        wrapper = DeepEPWrapper._instance
        if wrapper is None or wrapper.mode != DeepEPMode.LOW_LATENCY:
            raise RuntimeError(
                "PPU routed MXFP4 requires the engine's initialized DeepEP LL group"
            )
        cfg, comm = self.cfg, wrapper._config
        if (
            (
                comm.tp_size,
                comm.ep_size,
                comm.ep_rank,
                comm.hidden_size,
                comm.expert_num,
            )
            != (cfg.tp_size, cfg.ep_size, cfg.ep_rank, cfg.dim, cfg.n_routed_experts)
            or comm.moe_k != cfg.n_activated_experts
            or wrapper.ll_num_max_token_per_rank * max(cfg.tp_size, 1)
            < cfg.max_tokens_per_rank
        ):
            raise RuntimeError(
                "PPU MoE configuration differs from the engine-owned DeepEP group"
            )
        self._wrapper = wrapper
        return wrapper

    def forward(self, x, weights, indices, *, active_token_mask=None):
        wrapper = self._wrapper if self._wrapper is not None else self._bind_wrapper()
        return low_latency_mxfp4_moe(
            wrapper.buffer,
            x,
            weights,
            indices,
            (self._w13, self._s13),
            (self._w2, self._s2),
            num_experts=self.cfg.n_routed_experts,
            max_dispatch_tokens=wrapper.ll_num_max_token_per_rank,
            expected_m=self.expected_rows(x.shape[0]),
            swiglu_limit=self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None,
            output_dtype=self.output_dtype,
            active_token_mask=active_token_mask,
            gemm_tile=self._gemm_tile,
        )


def pad_topk(indices, weights, active_token_mask=None):
    """Pad unsupported widths (including V4's six) with inactive routes."""
    from rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding import pad_inactive_routes
    width = indices.shape[1]
    supported = (2, 4, 8, 16)
    if width in supported:
        if active_token_mask is None:
            return indices, weights
        return pad_inactive_routes(indices, weights, width, active_token_mask)
    target = next((n for n in supported if n > width), None)
    if target is None or width <= 0:
        raise ValueError("DeepEP topk must be in [1,16]")
    return pad_inactive_routes(indices, weights, target, active_token_mask)


def low_latency_mxfp4_moe(
    buffer,
    x,
    weights,
    indices,
    weight13,
    weight2,
    *,
    num_experts,
    max_dispatch_tokens,
    expected_m,
    swiglu_limit=None,
    output_dtype=torch.float32,
    active_token_mask=None,
    gemm_tile="auto",
):
    """Dispatch and combine every valid row, with no compact-prefix capacity.

    This adapter borrows the engine's existing buffer/communicator. The handle
    and payload stay alive through combine. Padded/inactive routes remain -1.
    BF16 output preserves the combine result for consumers that promote in
    registers; the FP32 default retains the existing adapter contract.
    """
    from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4_masked import (
        mxfp4_experts_masked,
        tensor_spans_overlap,
    )
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("PPU routed output must be FP32 or BF16")
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or indices.ndim != 2
        or weights.shape != indices.shape
        or x.shape[0] != indices.shape[0]
        or weights.dtype != torch.float32
        or indices.dtype != torch.int64
        or weights.device != x.device
        or indices.device != x.device
        or x.shape[0] > max_dispatch_tokens
    ):
        raise ValueError("Invalid DeepEP LL BF16 tokens / FP32 weights / int64 routes")
    indices, weights = pad_topk(indices, weights, active_token_mask)
    indices, weights = indices.contiguous(), weights.contiguous()
    expert_x, counts, handle, _, _ = buffer.low_latency_dispatch(
        x=x.contiguous(),
        topk_idx=indices,
        num_max_dispatch_tokens_per_rank=max_dispatch_tokens,
        num_experts=num_experts,
        use_fp8=False,
        use_mxfp4=True,
        mxfp4_scale_row_major=False,
        quant_size=32,
        async_finish=False,
        return_recv_hook=False,
    )
    if not isinstance(expert_x, tuple) or len(expert_x) != 2:
        raise RuntimeError("DeepEP LL MXFP4 dispatch must return (data, scale)")
    combine_slot = buffer.get_next_low_latency_combine_buffer(handle)
    # DeepEP versions may share arenas between dispatch and combine. Only
    # write w2 directly when the candidate cannot alias any live input.
    live_inputs = (*expert_x, *weight13, *weight2, counts)
    direct = not any(tensor_spans_overlap(combine_slot, t) for t in live_inputs)
    result = mxfp4_experts_masked(
        expert_x,
        weight13,
        weight2,
        counts,
        expected_m=expected_m,
        swiglu_limit=swiglu_limit,
        out=combine_slot if direct else None,
        gemm_tile=gemm_tile,
    )
    if not direct:
        if result.shape != combine_slot.shape:
            raise ValueError("DeepEP combine slot differs from the full expert output")
        combine_slot.copy_(result)
    combined, _, _ = buffer.low_latency_combine(
        x=combine_slot,
        topk_idx=indices,
        topk_weights=weights,
        handle=handle,
        zero_copy=True,
        async_finish=False,
        return_recv_hook=False,
    )
    if combined.dtype != torch.bfloat16:
        raise RuntimeError("PPU DeepEP LL combine must return BF16")
    return combined.to(output_dtype)
