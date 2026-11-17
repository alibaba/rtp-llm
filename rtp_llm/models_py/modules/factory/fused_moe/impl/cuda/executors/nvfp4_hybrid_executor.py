"""SM120 NVFP4 MoE: static b12x for small physical M, CUTLASS otherwise."""

import inspect
import logging
import os

import torch
from flashinfer import ActivationType
from flashinfer.fused_moe import cutlass_fused_moe

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
    B12xNvfp4MoE,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
    CutlassNvfp4Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_weights import (
    Nvfp4WeightViews,
)

logger = logging.getLogger(__name__)
_MAX_B12X_ROWS = 16  # Largest physical M validated in Decode serving.


def decode_tuning(quant_config):
    max_rows = getattr(quant_config, "decode_b12x_max_rows", 0)
    profiles = getattr(quant_config, "decode_cutlass_profile_ids", None)
    if type(max_rows) is not int or not 0 <= max_rows <= _MAX_B12X_ROWS:
        raise ValueError(
            f"decode_b12x_max_rows must be an integer in [0, {_MAX_B12X_ROWS}]"
        )
    if profiles is not None:
        if (
            not isinstance(profiles, (tuple, list))
            or len(profiles) != 2
            or any(type(value) is not int or value < -1 for value in profiles)
        ):
            raise ValueError(
                "decode_cutlass_profile_ids requires two integer tactic IDs >= -1"
            )
        profiles = list(profiles)
    return max_rows, profiles


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
        checker.check(config.hidden_size > 0 and config.hidden_size % 128 == 0)
        # The pinned static b12x ABI consumes tile-aligned block16 scale views.
        checker.check(config.moe_inter_dim >= 256 and config.moe_inter_dim % 128 == 0)
        checker.check(config.n_local_experts > 0)
        checker.check(0 < config.moe_k <= config.expert_num)
        checker.check(config.moe_w1_layout == "up_gate")
        if checker.all_passed():
            rows, profiles = decode_tuning(config.model_config.quant_config)
            if not checker.check(bool(rows or profiles is not None)):
                return
            checker.check(
                profiles is None
                or "profile_ids" in inspect.signature(cutlass_fused_moe).parameters
            )
            checker.check(
                rows == 0 or (pc.ep_size == 2 and B12xNvfp4MoE.is_available())
            )

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        self.max_b12x_rows, self.cutlass_profiles = decode_tuning(
            config.model_config.quant_config
        )
        if (
            self.cutlass_profiles is not None
            and "profile_ids" not in inspect.signature(cutlass_fused_moe).parameters
        ):
            raise RuntimeError(
                "Configured NVFP4 tactics require FlashInfer profile_ids support. "
                "Use a compatible build or omit decode_cutlass_profile_ids."
            )
        prepared = Nvfp4WeightViews(config, quant_config, weights)
        self.w1, self.w2 = prepared.w1, prepared.w2
        self.quant_scales = prepared.cutlass_quant_scales()
        self.ep_size, self.ep_rank = prepared.ep_size, prepared.ep_rank
        self.tp_size, self.tp_rank = prepared.tp_size, prepared.tp_rank
        self.hidden_size = self.w2.shape[1]
        self.top_k = config.moe_k
        self.b12x = (
            B12xNvfp4MoE(
                self.w1,
                self.w2,
                self.quant_scales,
                self.ep_rank,
                config.model_config,
                self.top_k,
            )
            if self.max_b12x_rows
            else None
        )

        logger.info(
            "SM120 NVFP4 hybrid: b12x M=1..%d, CUTLASS profiles=%s; E=%d K=%d N=%d topk=%d",
            self.max_b12x_rows,
            self.cutlass_profiles,
            self.w1.shape[0],
            self.hidden_size,
            self.w2.shape[2] * 16,
            self.top_k,
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
        if self.b12x is not None and 0 < x.shape[0] <= self.max_b12x_rows:
            output = self.b12x(x, ids, router)
        else:
            tactics = (
                {}
                if self.cutlass_profiles is None
                else {"profile_ids": self.cutlass_profiles}
            )
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
                **tactics,
            )
            if isinstance(output, list):
                output = output[0]
        return CombineForwardPayload(fused_expert_output=output)
