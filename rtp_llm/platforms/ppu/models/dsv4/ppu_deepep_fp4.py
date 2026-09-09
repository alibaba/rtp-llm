"""Instance-selected V4 routed experts on the engine's PPU DeepEP LL group."""

import torch
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import RoutedExpertsStrategy


def prepare_routed_mxfp4_weights(cfg, layer_weights):
    """Preserve EP-sliced checkpoint nibbles; prepare scale pairs once."""
    from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import prepare_fp4_weight_scale_mxfp4
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
    if not w1.is_cuda or torch.cuda.get_device_name(w1.device) != "ZW-M890P":
        raise RuntimeError("PPU routed MXFP4 requires ZW-M890P weights")
    result = (
        torch.cat((w1.view(torch.uint8), w3.view(torch.uint8)), dim=1).contiguous(),
        prepare_fp4_weight_scale_mxfp4(torch.cat((s1, s3), dim=1).contiguous()),
        w2.view(torch.uint8).contiguous(),
        prepare_fp4_weight_scale_mxfp4(s2.contiguous()),
    )
    for key in keys:
        layer_weights.pop(key)
    return result


class PpuDeepEPFP4Strategy(RoutedExpertsStrategy):
    """TP1, DP/EP routed compute; selection is explicit, never process-global."""

    name = "ppu_deepep_fp4"

    @classmethod
    def can_handle(cls, cfg):
        return (
            cfg.tp_size == 1
            and cfg.ep_size > 1
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
        self, cfg, *, expected_m_policy="capacity", output_dtype=torch.float32
    ):
        super().__init__(cfg)
        if output_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("PPU routed output must be FP32 or BF16")
        self.output_dtype = output_dtype
        if expected_m_policy != "capacity":
            raise ValueError("PPU MoE expected rows policy must be capacity")
        if not self.can_handle(cfg):
            raise ValueError(
                "PPU DeepEP MXFP4 requires TP1 with compatible EP-local experts"
            )
        self._wrapper = None
        self._expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank * cfg.ep_size * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )

    def expected_rows(self, num_tokens):
        """Host launch hint only; never truncate the buffer or device counts."""
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
            or wrapper.ll_num_max_token_per_rank < cfg.max_tokens_per_rank
        ):
            raise RuntimeError(
                "PPU MoE configuration differs from the engine-owned DeepEP group"
            )
        self._wrapper = wrapper
        return wrapper

    def forward(self, x, weights, indices):
        from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import (
            low_latency_mxfp4_moe,
        )

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
        )
