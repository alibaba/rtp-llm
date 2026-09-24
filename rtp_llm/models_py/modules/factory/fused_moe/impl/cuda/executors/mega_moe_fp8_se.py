"""FP8 MegaMoE + gated shared expert. Selected by ``mega_moe_fp8_se``."""

from __future__ import annotations

import inspect
import logging
from typing import Dict

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertGatePayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _mega_output_capacity,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
    mega_moe_fp8_available,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_impl import (
    mega_moe_fp8_impl,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
    expand_fp8_scale,
    prepare_mega_moe_fp8_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.group import (
    get_validated_world_ep_group,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
    _get_or_create_mega_fp8_se_buf,
    _get_or_create_mega_se_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_input_packer import (
    get_mega_moe_se_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs import (
    ensure_shared_gate_capacity,
)
from rtp_llm.utils.model_weight import W


def mega_moe_fp8_se_available():
    if not mega_moe_fp8_available():
        return False
    try:
        from deep_gemm import mega_fp8
    except ImportError:
        return False
    return (
        "shared_expert_gates" in inspect.signature(mega_fp8.fp8_fp8_mega_moe).parameters
        and getattr(mega_fp8, "RTP_GATED_SHARED_EXPERT_SEMANTICS", 0) == 1
    )


class MegaMoeFp8SEExecutor(MegaMoeFp8Executor):
    gated_shared_expert_requested = True
    _num_shared_experts = 1

    @classmethod
    def check_conditions(cls, checker, config):
        super().check_conditions(checker, config)
        # Fused shared experts need complete weights on each rank. Ordinary
        # mega_moe_fp8 keeps TP-sharded shared experts in GenericMoeLayer.
        checker.check(config.tp_size == 1)
        checker.check(getattr(config, "n_shared_experts", 0) == 1)
        checker.check(bool(getattr(config, "has_shared_expert_gate", False)))
        checker.check(mega_moe_fp8_se_available())

    def __init__(self, config, quant_config, weights):
        mega_moe_fp8_impl(shared_expert_gates=True)
        from deep_gemm import mega_fp8

        if (
            "shared_expert_gates"
            not in inspect.signature(mega_fp8.fp8_fp8_mega_moe).parameters
        ):
            raise RuntimeError(
                "Gated MegaMoE requires a DeepGEMM build with shared_expert_gates"
            )
        if getattr(mega_fp8, "RTP_GATED_SHARED_EXPERT_SEMANTICS", 0) != 1:
            raise RuntimeError(
                "Gated MegaMoE requires RTP BF16/group-128 shared-expert semantics v1"
            )
        super().__init__(config, quant_config, weights)

    def setup_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        impl = mega_moe_fp8_impl(shared_expert_gates=True)
        if not mega_moe_fp8_available():
            raise RuntimeError(
                "mega_moe_fp8_se requires SM10x and DeepGEMM mega_fp8 support"
            )
        self.uses_shared_expert_gates = False
        self.shared_l1 = self.shared_l2 = None
        config = self.cfg
        self.l1, self.l2 = prepare_mega_moe_fp8_weights(
            weights[W.moe_w1],
            weights[W.moe_s1],
            weights[W.moe_w2],
            weights[W.moe_s2],
            config.moe_inter_dim,
            config.moe_w1_layout,
        )
        del weights[W.moe_s1], weights[W.moe_s2]
        self._mega_l1_w = self.l1[0]
        self._mega_group = get_validated_world_ep_group(config, dist)
        self._mega_buf = _get_or_create_mega_fp8_se_buf(
            group=self._mega_group,
            num_experts=config.expert_num,
            num_max_tokens_per_rank=max(config.max_tokens_per_rank, 1),
            num_topk=config.moe_k,
            hidden=config.hidden_size,
            intermediate_hidden=config.moe_inter_dim,
            num_shared_experts=int(self._num_shared_experts),
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        self._mega_y = _get_or_create_mega_se_output(
            _mega_output_capacity(self._mega_buf, config.max_tokens_per_rank),
            config.hidden_size,
            torch.bfloat16,
            self.l1[0].device,
        )
        ensure_shared_gate_capacity(
            self.l1[0].device, max(int(config.max_tokens_per_rank), 1)
        )
        self._input_packer = get_mega_moe_se_input_packer()
        self._maybe_warmup_jit_once()
        logging.info(
            "MegaMoE FP8-SE weights prepared during model construction: impl=%s, experts=%d, "
            "hidden=%d, intermediate=%d, max_tokens_per_rank=%d, shared=%d",
            impl,
            config.n_local_experts,
            config.hidden_size,
            config.moe_inter_dim,
            config.max_tokens_per_rank,
            self._num_shared_experts,
        )

    def configure_gated_shared_expert(self, shared_expert):
        from deep_gemm import mega_fp8

        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )

        linears = [
            getattr(p, "_deepgemm_linear", p)
            for p in (shared_expert.up_proj, shared_expert.down_proj)
        ]
        if not all(
            isinstance(p, CudaFp8DeepGEMMLinear) and p.scale_ue8m0 and p.bias is None
            for p in linears
        ):
            raise ValueError(
                "Gated MegaMoE needs bias-free FP8 UE8M0 shared-expert linears"
            )
        up, down = linears
        inter = self.config.moe_inter_dim
        hidden = self.config.hidden_size
        if tuple(up.weight.shape) != (2 * inter, hidden) or tuple(
            down.weight.shape
        ) != (hidden, inter):
            raise ValueError(
                "Gated MegaMoE requires one shared expert with matching intermediate width"
            )
        self.shared_l1, self.shared_l2 = mega_fp8.transform_weights_for_mega_moe_fp8(
            (up.weight, expand_fp8_scale(up.weight_scales, 2 * inter, hidden)),
            (down.weight, expand_fp8_scale(down.weight_scales, hidden, inter)),
        )
        self.shared_l1, self.shared_l2 = tuple(
            (weight, scale.transpose(-1, -2).contiguous().transpose(-1, -2))
            for weight, scale in (self.shared_l1, self.shared_l2)
        )
        self.includes_shared_expert = True
        self.uses_shared_expert_gates = True
        self._maybe_warmup_jit_once()
        logging.info(
            "MegaMoE gated shared expert configured: intermediate=%d, DeepGEMM=%s",
            inter,
            mega_fp8.__file__,
        )

    def _block_m(self, tokens: int) -> int:
        from deep_gemm import mega_fp8

        buf = self._mega_buf
        return int(
            mega_fp8.get_block_m_for_mega_moe_fp8(
                self.config.ep_size,
                self.config.expert_num,
                buf.num_max_tokens_per_rank,
                int(tokens),
                self.config.moe_k,
            )
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not self.uses_shared_expert_gates:
            return
        super()._maybe_warmup_jit_once()

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        if not token_counts:
            return
        self._jit_warmup_shared_gates = torch.ones(
            max(token_counts),
            dtype=torch.float32,
            device=self.l1[0].device,
        )
        try:
            super().warmup_jit(token_counts)
        finally:
            del self._jit_warmup_shared_gates

    def _shared_kwargs(self, extra, tokens: int, device: torch.device):
        extra = dict(extra or {})
        shared_gates = extra.pop("shared_expert_gates", None)
        if shared_gates is None and hasattr(self, "_jit_warmup_shared_gates"):
            shared_gates = self._jit_warmup_shared_gates[:tokens]
        if extra:
            raise ValueError(
                "mega_moe_fp8_se only accepts shared_expert_gates extra args"
            )
        if not self.uses_shared_expert_gates:
            raise RuntimeError("Gated shared expert was requested but not configured")
        if shared_gates is None:
            raise ValueError("mega_moe_fp8_se requires shared_expert_gates")
        if (
            shared_gates.shape != (tokens,)
            or shared_gates.dtype != torch.float32
            or shared_gates.device != device
            or not shared_gates.is_contiguous()
        ):
            raise ValueError("Shared gates must be contiguous CUDA FP32 [local_tokens]")
        return dict(
            shared_l1_weights=self.shared_l1,
            shared_l2_weights=self.shared_l2,
            shared_expert_gates=shared_gates,
        )

    def forward(self, x, weights, indices, extra_expert_args=None):
        """Return BF16 routed+shared output; participate even for local T=0."""
        tokens = x.size(0)
        self._validate_capacity(tokens)
        block_m = self._block_m(tokens)
        self._input_packer.pack(x, weights, indices, self._mega_buf, tokens, block_m)
        y = self._mega_y[:tokens]
        self._launch(
            y,
            tokens,
            x.device,
            diagnostic_inputs={"x": x, "weights": weights, "indices": indices},
            **self._shared_kwargs(extra_expert_args, tokens, x.device),
        )
        return y

    def forward_gate_pack(
        self,
        x: torch.Tensor,
        gate_payload: ExpertGatePayload,
        extra_expert_args=None,
    ):
        if not self.supports_gate_pack:
            raise RuntimeError(
                "MegaMoE FP8-SE fused gate packing requires the fused packer"
            )
        tokens = x.size(0)
        self._validate_capacity(tokens)
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack import (
            stage_mega_moe_se_shared_l1_scales,
        )

        buf = self._mega_buf
        fused_pack_mega_moe_gate_inputs(
            x,
            gate_payload.scores,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
            topk=gate_payload.topk,
            score_func=gate_payload.score_func,
            route_scale=gate_payload.route_scale,
            norm_eps=gate_payload.norm_eps,
            bias=gate_payload.bias,
            input_ids=gate_payload.input_ids,
            tid2eid=gate_payload.tid2eid,
        )
        stage_mega_moe_se_shared_l1_scales(
            buf.x_sf[:tokens],
            buf.shared_l1_acts_sf,
            tokens,
            self._block_m(tokens),
        )
        y = self._mega_y[:tokens]
        self._launch(
            y,
            tokens,
            x.device,
            diagnostic_inputs={
                "x": x,
                "scores": gate_payload.scores,
                "topk": gate_payload.topk,
                "score_func": gate_payload.score_func,
                "route_scale": gate_payload.route_scale,
                "norm_eps": gate_payload.norm_eps,
                "bias": gate_payload.bias,
                "input_ids": gate_payload.input_ids,
                "tid2eid": gate_payload.tid2eid,
            },
            **self._shared_kwargs(extra_expert_args, tokens, x.device),
        )
        return y, buf.topk_weights[:tokens], buf.topk_idx[:tokens]

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        if activation.lower() not in ("silu", "siglu", "swiglu"):
            raise ValueError("mega_moe_fp8_se supports SwiGLU only")
        if (
            expert_map is not None
            or a2_scale is not None
            or apply_router_weight_on_input
        ):
            raise ValueError(
                "mega_moe_fp8_se does not support expert remapping or externally scaled activations"
            )
        if getattr(payload, "gate_payload", None) is not None:
            output, topk_weights, topk_ids = self.forward_gate_pack(
                payload.expert_x, payload.gate_payload, extra_expert_args
            )
            payload.expert_topk_weights = topk_weights
            payload.expert_topk_ids = topk_ids
            return CombineForwardPayload(
                fused_expert_output=self._restore_output_dtype(payload, output)
            )
        topk_weights = getattr(payload, "expert_topk_weights", None)
        topk_ids = getattr(payload, "expert_topk_ids", None)
        if topk_weights is None or topk_ids is None:
            raise ValueError("mega_moe_fp8_se requires routed top-k weights and ids")
        return CombineForwardPayload(
            fused_expert_output=self._restore_output_dtype(
                payload,
                self.forward(
                    payload.expert_x, topk_weights, topk_ids, extra_expert_args
                ),
            )
        )
