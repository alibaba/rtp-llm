from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.utils.model_weight import W


def _relax_b12x_cuda_version_gate() -> None:
    """Relax flashinfer's CUDA>=13 gate for the b12x NVFP4 MoE kernel."""
    from flashinfer.jit import cpp_ext

    real_version = cpp_ext.get_cuda_version()
    if real_version.major >= 13:
        return  # native gate already passes, no patch needed
    if (real_version.major, real_version.minor) >= (12, 9):
        # Same type as the real object so .major / str() behave identically.
        fake_version = type(real_version)("13.0")
        cpp_ext.get_cuda_version = lambda: fake_version


class B12xFp4Executor(FusedMoeExpertExecutor):
    """flashinfer b12x CuTe DSL fused NVFP4 MoE executor for sm_120/sm_121."""

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.B12X_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(
            resolver.has_quantization(config)
            and resolver.get_quant_method(config) == "modelopt_fp4"
        )
        checker.check(is_sm12x())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        _relax_b12x_cuda_version_gate()
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe import b12x_fused_moe

        self._b12x_fused_moe = b12x_fused_moe

        self.w1 = weights.get(W.moe_w1, None)  # [E, 2*I, H//2] uint8, up-first
        self.w2 = weights.get(W.moe_w2, None)  # [E, H, I//2] uint8
        w1_sf = weights.get(W.moe_s1, None)  # fp8_e4m3, swizzled blockscale
        w2_sf = weights.get(W.moe_s2, None)  # fp8_e4m3, swizzled blockscale

        w1_scale_2 = weights.get(W.moe_w1_s2, None)  # [E] weight_scale_2 (w13)
        w2_scale_2 = weights.get(W.moe_w2_s2, None)  # [E] weight_scale_2 (w2)

        assert (
            self.w1 is not None and self.w2 is not None
        ), "b12x FP4 needs moe_w1/moe_w2"
        assert w1_sf is not None and w2_sf is not None, "b12x FP4 needs moe_s1/moe_s2"
        assert (
            w1_scale_2 is not None and w2_scale_2 is not None
        ), "b12x FP4 needs weight_scale_2"

        self.num_experts = config.expert_num
        self.top_k = config.moe_k

        E, two_i, h_half = self.w1.shape
        assert two_i % 2 == 0, f"w13 rows must be 2*I, got {two_i}"
        self.intermediate_size = two_i // 2  # I (after any TP split)
        self.hidden_size = self.w2.shape[1]  # H
        assert (
            h_half * 2 == self.hidden_size
        ), f"w13 last dim {h_half} * 2 must equal hidden {self.hidden_size}"

        # Fold weight_scale_2 into the (already swizzled) block scale factors so
        # the kernel's per-block scales carry the full weight scale and the global
        # alphas can be 1 (a per-expert scalar multiply commutes with the swizzle
        # permutation). Then convert to the 6D MMA layout the kernel consumes;
        # m = weight rows (2*I for w13, H for w2), k = contraction dim.
        w1_sf_folded = (
            w1_sf.to(torch.float32) * w1_scale_2.reshape(E, 1, 1).to(torch.float32)
        ).to(torch.float8_e4m3fn)
        w2_sf_folded = (
            w2_sf.to(torch.float32) * w2_scale_2.reshape(E, 1, 1).to(torch.float32)
        ).to(torch.float8_e4m3fn)
        self.w1_sf_mma = convert_sf_to_mma_layout(
            w1_sf_folded.reshape(-1).contiguous(),
            m=two_i,
            k=self.hidden_size,
            num_groups=E,
            sf_vec_size=16,
        )
        self.w2_sf_mma = convert_sf_to_mma_layout(
            w2_sf_folded.reshape(-1).contiguous(),
            m=self.hidden_size,
            k=self.intermediate_size,
            num_groups=E,
            sf_vec_size=16,
        )

        # Global scales are 1: weight scale is folded into the block factors, and
        # activation/intermediate quantization relies on per-block e4m3 scales.
        device = self.w1.device
        self.w1_alpha = torch.ones(E, dtype=torch.float32, device=device)
        self.w2_alpha = torch.ones(E, dtype=torch.float32, device=device)
        self.fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)

    @property
    def local_num_experts(self) -> int:
        assert self.w1 is not None
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        expert_x = payload.expert_x
        assert expert_x is not None
        assert (
            expert_x.dtype is torch.bfloat16
        ), f"b12x consumes bf16 activations directly, got {expert_x.dtype}"
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        act = activation.lower()
        assert act in (
            "silu",
            "swiglu",
            "siglu",
        ), f"b12x MoE supports gated SiLU/SwiGLU only, got {activation}"

        output = self._b12x_fused_moe(
            x=expert_x,  # [T, H] bf16
            w1_weight=self.w1,
            w1_weight_sf=self.w1_sf_mma,
            w2_weight=self.w2,
            w2_weight_sf=self.w2_sf_mma,
            token_selected_experts=payload.expert_topk_ids,
            token_final_scales=payload.expert_topk_weights,
            num_experts=self.num_experts,
            top_k=payload.expert_topk_ids.size(-1),
            w1_alpha=self.w1_alpha,
            w2_alpha=self.w2_alpha,
            fc2_input_scale=self.fc2_input_scale,
            output_dtype=torch.bfloat16,
            activation="silu",
            quant_mode="nvfp4",
            source_format="modelopt",
        )

        return CombineForwardPayload(fused_expert_output=output)
