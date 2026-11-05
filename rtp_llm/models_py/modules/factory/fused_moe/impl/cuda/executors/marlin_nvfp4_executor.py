"""Qwen4 weight adaptation and factory admission for NVFP4 Marlin."""

import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import Nvfp4MarlinMoE
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)


def _linear_scales(swizzled, experts, rows, cols):
    return (
        swizzled.view(torch.float8_e4m3fn)
        .reshape(experts, rows // 128, cols // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows, cols)
    )


class Qwen4DecodeMarlinExecutor(FusedMoeExpertExecutor):
    """Adapt prepared Qwen4 NVFP4 weights to the common Marlin kernel.

    Factory admission remains limited to qwen4_exp and qwen4_exp_mtp; other
    models require separate validation before their admission is enabled.
    """

    @classmethod
    def executor_type(cls):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

        return ExecutorType.CUTLASS_NVFP4

    @property
    def topk_ids_dtype(self):
        return torch.int32

    @property
    def local_num_experts(self):
        return self.moe.experts

    def __init__(self, native):
        from rtp_llm.utils.model_weight import W

        super().__init__(native.config, native.quant_config, native.weights)
        self.ep_size, self.ep_rank = native.ep_size, native.ep_rank
        self.tp_size, self.tp_rank = native.tp_size, native.tp_rank
        q1, q2 = native.w1.view(torch.uint8), native.w2.view(torch.uint8)
        experts, both_n, packed_k = q1.shape
        n, k = both_n // 2, packed_k * 2
        owner = self.weights[W.moe_w1]
        identity = (
            self.ep_size,
            self.ep_rank,
            q2.data_ptr(),
            native.quant_scales[1].data_ptr(),
            native.quant_scales[4].data_ptr(),
        )
        prior = getattr(owner, "_qwen4_marlin_layout", None)
        if prior is not None:
            if prior[:5] != identity:
                raise ValueError("Changing packed Qwen MoE weights requires restart")
            moe = object.__new__(Nvfp4MarlinMoE)
            moe.experts, moe.n, moe.k = experts, n, k
            moe._one_row_sm120 = (k, n) == (
                2560,
                640,
            ) and torch.cuda.get_device_capability(q1.device) == (12, 0)
            moe.first = experts * self.ep_rank
            moe.w1 = q1.view(torch.int32).reshape(experts, k // 16, both_n * 2)
            moe.w2 = q2.view(torch.int32).reshape(experts, n // 16, k * 2)
            moe.s1 = (
                native.quant_scales[1]
                .view(torch.float8_e4m3fn)
                .reshape(experts, k // 16, both_n)
            )
            moe.s2 = (
                native.quant_scales[4]
                .view(torch.float8_e4m3fn)
                .reshape(experts, n // 16, k)
            )
            moe.g1, moe.g2 = prior[5:]
            moe.workspaces = {}
            self.moe = moe
            return
        sf1 = _linear_scales(native.quant_scales[1], experts, both_n, k // 16)
        sf2 = _linear_scales(native.quant_scales[4], experts, k, n // 16)
        g1, g2 = native.weights[W.moe_w1_s2], native.weights[W.moe_w2_s2]
        for scale in (sf1, sf2, g1, g2):
            if not bool(torch.isfinite(scale.float()).all()) or bool(
                (scale.float() < 0).any()
            ):
                raise ValueError("Marlin NVFP4 requires finite nonnegative scales")
        moe = Nvfp4MarlinMoE(q1, sf1, g1, q2, sf2, g2, self.ep_size, self.ep_rank)
        destinations = (q1, q2, native.quant_scales[1], native.quant_scales[4])
        converted = (moe.w1, moe.w2, moe.s1, moe.s2)
        for target, source in zip(destinations, converted):
            if (
                target.numel() * target.element_size()
                != source.numel() * source.element_size()
            ):
                raise ValueError("Marlin repack must preserve weight storage size")
        packed = []
        for target, source in zip(destinations, converted):
            target.view(torch.uint8).reshape(-1).copy_(
                source.view(torch.uint8).reshape(-1)
            )
            packed.append(target.view(source.dtype).reshape(source.shape))
        moe.w1, moe.w2, moe.s1, moe.s2 = packed
        # Keep only tiny global scales on the source Tensor. Holding the MoE
        # here would create a cycle through its packed views and their owner.
        owner._qwen4_marlin_layout = (*identity, moe.g1, moe.g2)
        self.moe = moe

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        if activation.lower() not in ("silu", "swiglu", "siglu"):
            raise ValueError("Marlin NVFP4 requires SwiGLU")
        if (
            apply_router_weight_on_input
            or payload.expert_ids_are_local
            or expert_map is not None
        ):
            raise ValueError(
                "Marlin NVFP4 requires global IDs and output router weighting"
            )
        x = payload.expert_x
        if x.numel() == 0:
            output = torch.empty_like(x)
        else:
            output = self.moe(
                x,
                payload.expert_topk_ids.to(torch.int32),
                payload.expert_topk_weights.to(torch.float32),
            )
        return CombineForwardPayload(fused_expert_output=output)


class Qwen4MarlinExecutor(Qwen4DecodeMarlinExecutor):
    """Explicit Marlin selection, currently restricted to Qwen4 models.

    Common placement fixes the dependency direction; it does not expand
    admission beyond qwen4_exp and qwen4_exp_mtp without model validation.
    """

    @classmethod
    def executor_type(cls):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

        return ExecutorType.MARLIN_NVFP4

    @classmethod
    def check_conditions(cls, checker, config):
        from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant = config.model_config.quant_config
        checker.check(resolver.is_bf16(config))
        checker.check(resolver.get_quant_method(config) == "modelopt_fp4")
        checker.check(config.model_config.model_type in ("qwen4_exp", "qwen4_exp_mtp"))
        checker.check(
            quant is not None
            and getattr(quant, "model_recipe", None) == QWEN4_FP8_NVFP4_RECIPE
            and not quant.is_quanted()
        )
        checker.check(Nvfp4MarlinMoE.is_supported())
        checker.check(config.tp_size == config.ep_size)
        checker.check(config.parallelism_config.get_ffn_tp_size() == 1)
        checker.check(config.moe_config.fp4_moe_op in ("auto", "marlin"))
        checker.check(not config.has_redundant_experts)
        checker.check(not config.model_config.eplb_config.enable_eplb())

    def __init__(self, config, quant_config, weights):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
            CutlassNvfp4Executor,
        )

        # Reuse the prepared scale views and in-place repacking contract. The
        # CUTLASS executor is only a weight adapter here; its kernel is not run.
        super().__init__(CutlassNvfp4Executor(config, quant_config, weights))
