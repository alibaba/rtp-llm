"""SM120 CUTLASS NVFP4 MoE with BF16 input and fused activation quantization."""

import torch
from flashinfer import ActivationType
from flashinfer.fused_moe import cutlass_fused_moe

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_parallelism import (
    use_nvfp4_tensor_parallel,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_weights import (
    Nvfp4WeightViews,
)


def _prefill_library(config):
    return (
        getattr(config, "prefill_library", None),
        getattr(config, "prefill_library_sha256", None),
    )


class CutlassNvfp4Executor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_NVFP4

    @classmethod
    def check_conditions(cls, checker, config):
        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(resolver.get_quant_method(config) == "modelopt_fp4")
        checker.check(
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
        )
        tensor_parallel = use_nvfp4_tensor_parallel(config)
        checker.check(
            config.parallelism_config.get_ffn_tp_size() == 1 or tensor_parallel
        )
        checker.check(
            config.moe_config.fp4_moe_op == "cutlass"
            or (
                config.moe_config.fp4_moe_op == "auto"
                and config.moe_config.moe_strategy == "nvfp4_cutlass"
            )
        )
        quant_config = config.model_config.quant_config
        checker.check(quant_config is not None and not quant_config.is_quanted())
        checker.check(not config.has_redundant_experts)
        checker.check(not config.model_config.eplb_config.enable_eplb())
        checker.check(config.tp_size == config.ep_size or tensor_parallel)

    @property
    def topk_ids_dtype(self):
        return torch.int32

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        prepared = Nvfp4WeightViews(config, quant_config, weights)
        self.w1, self.w2 = prepared.w1, prepared.w2
        self.quant_scales = prepared.quant_scales
        self.ep_size, self.ep_rank = prepared.ep_size, prepared.ep_rank
        self.tp_size, self.tp_rank = prepared.tp_size, prepared.tp_rank
        self.sm120_prefill = None
        backend_config = getattr(config.model_config, "quant_config", None)
        library, library_sha256 = _prefill_library(backend_config)
        if (
            library
            and self.w1.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(self.w1.device) == (12, 0)
        ):
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_nvfp4_prefill import (
                Sm120PrefillMoE,
            )

            self.sm120_prefill = Sm120PrefillMoE(
                library, library_sha256, self.w1.device
            )

    @property
    def local_num_experts(self):
        return self.w1.shape[0]

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
            raise ValueError("CUTLASS NVFP4 requires SwiGLU")
        if (
            apply_router_weight_on_input
            or payload.expert_ids_are_local
            or expert_map is not None
        ):
            raise ValueError(
                "CUTLASS NVFP4 requires global expert IDs and output router weighting"
            )
        expert_x = payload.expert_x
        if (
            self.sm120_prefill is not None
            and expert_x.ndim == 2
            and expert_x.dtype == torch.bfloat16
            and expert_x.is_contiguous()
            and expert_x.shape[0] >= 128
            and expert_x.shape[1] == 2560
            and self.w1.ndim == self.w2.ndim == 3
            and self.w1.shape[0] == self.w2.shape[0]
            and self.w1.shape[2] * 16 == expert_x.shape[1]
            and self.w2.shape[1] == expert_x.shape[1]
            and self.w1.shape[1] == 32 * self.w2.shape[2]
            and not torch.cuda.is_current_stream_capturing()
        ):
            output = self.sm120_prefill(
                expert_x,
                payload.expert_topk_ids.to(torch.int32),
                payload.expert_topk_weights.to(torch.float32),
                self.w1,
                self.w2,
                self.quant_scales,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
            )
            return CombineForwardPayload(fused_expert_output=output)
        output = cutlass_fused_moe(
            input=expert_x,
            input_sf=None,
            token_selected_experts=payload.expert_topk_ids.to(torch.int32),
            token_final_scales=payload.expert_topk_weights.to(torch.float32),
            fc1_expert_weights=self.w1,
            fc2_expert_weights=self.w2,
            output_dtype=payload.expert_x_origin_dtype or torch.bfloat16,
            quant_scales=self.quant_scales,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            activation_type=ActivationType.Swiglu,
            enable_pdl=False,
        )
        if isinstance(output, list):
            output = output[0]
        return CombineForwardPayload(fused_expert_output=output)
