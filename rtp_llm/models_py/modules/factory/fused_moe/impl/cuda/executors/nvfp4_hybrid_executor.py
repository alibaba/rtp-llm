"""SM120 NVFP4 MoE: b12x micro at physical M=2, CUTLASS otherwise."""

import logging
import os

import torch
from flashinfer import ActivationType
from flashinfer.fused_moe import cutlass_fused_moe

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
    CutlassNvfp4Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_decode_069 import (
    LockedDecodeMoE,
    uses_locked_wheel,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_weights import (
    Nvfp4WeightViews,
)

logger = logging.getLogger(__name__)


def _locked_recipe(config):
    # FlashInfer tactic IDs are binary- and geometry-specific.
    return (
        uses_locked_wheel()
        and getattr(config.model_config.quant_config, "model_recipe", None)
        == QWEN4_FP8_NVFP4_RECIPE
        and config.parallelism_config.ep_size == 2
        and (
            config.n_local_experts,
            config.hidden_size,
            config.moe_inter_dim,
            config.moe_k,
        )
        == (256, 2560, 640, 10)
    )


class Nvfp4HybridExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.NVFP4_HYBRID

    @property
    def topk_ids_dtype(self):
        # Both backends consume int32; preserve the fused router's admission.
        return torch.int32

    @classmethod
    def check_conditions(cls, checker, config):
        CutlassNvfp4Executor.check_conditions(checker, config)
        pc = config.parallelism_config
        if not checker.check(
            os.getenv("RTP_NVFP4_HYBRID_MOE", "1") == "1"
            and str(pc.role_type).rsplit(".", 1)[-1] == "DECODE"
        ):
            return
        checker.check(
            torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
        )
        checker.check(pc.tp_size == pc.ep_size and pc.dp_size == pc.pp_size == 1)
        checker.check(pc.get_ffn_tp_size() == 1)
        checker.check(config.moe_w1_layout == "up_gate")
        if checker.all_passed() and checker.check(_locked_recipe(config)):
            checker.check(config.model_config.quant_config.decode_b12x_max_rows > 0)

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        prepared = Nvfp4WeightViews(config, quant_config, weights)
        self.w1, self.w2 = prepared.w1, prepared.w2
        self.quant_scales = prepared.cutlass_quant_scales()
        self.ep_size, self.ep_rank = prepared.ep_size, prepared.ep_rank
        self.tp_size, self.tp_rank = prepared.tp_size, prepared.tp_rank
        self.hidden_size = self.w2.shape[1]
        self.top_k = config.moe_k
        self.locked_decode = LockedDecodeMoE(
            self.w1,
            self.w2,
            self.quant_scales,
            self.ep_rank,
            config.model_config,
            self.top_k,
            config.model_config.quant_config.decode_b12x_max_rows >= 2,
        )
        logger.info(
            "SM120 FlashInfer 0.6.9: M2 b12x=%s, CUTLASS profiles=%s",
            self.locked_decode.micro,
            self.locked_decode.profile_table_matches,
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
            raise ValueError("NVFP4 hybrid requires SwiGLU")
        if (
            apply_router_weight_on_input
            or payload.expert_ids_are_local
            or expert_map is not None
        ):
            raise ValueError(
                "NVFP4 hybrid requires global expert IDs and output router weighting"
            )
        x = payload.expert_x
        if not (
            x.ndim == 2
            and x.shape[1] == self.hidden_size
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and payload.expert_topk_ids.shape == (x.shape[0], self.top_k)
            and payload.expert_topk_weights.shape == (x.shape[0], self.top_k)
        ):
            raise ValueError(
                "NVFP4 hybrid payload dimensions or BF16 layout do not match its weights"
            )
        ids = payload.expert_topk_ids.to(torch.int32).contiguous()
        router = payload.expert_topk_weights.to(torch.float32).contiguous()
        # M is the executor's physical input rows, including graph padding.
        # The router's EP partition and HTTP concurrency do not determine M.
        if payload.expert_x_origin_dtype in (None, torch.bfloat16):
            output = self.locked_decode.execute(x, ids, router)
            if output is not None:
                return CombineForwardPayload(fused_expert_output=output)
        output = cutlass_fused_moe(
            input=x,
            input_sf=None,
            token_selected_experts=ids,
            token_final_scales=router,
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
