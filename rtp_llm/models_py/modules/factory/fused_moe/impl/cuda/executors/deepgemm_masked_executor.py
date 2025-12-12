from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_masked_post_quant_fwd,
    silu_mul_bf16_deep_gemm_masked,
    silu_mul_fp8_quant_deep_gemm_masked,
)
from rtp_llm.utils.model_weight import W


class DeepGemmMaskedExecutor(FusedMoeExpertExecutor):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if DeepGemmMaskedExecutor can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        print(
            "ccc:",
            has_deep_gemm(),
            resolver.is_bf16(config),
            resolver.get_quant_method(config),
            flush=True,
        )
        checker.check(has_deep_gemm())
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize the DeepGemmMaskedExecutor.
        Args:
            config: Model configuration.
            weights: Dictionary containing model weights.
            quant_config: Quantization configuration.
        """
        super().__init__(quant_config=quant_config)
        self._config = config
        self._weights = weights
        # init weights
        self._w1 = self._weights.get(W.moe_w1, None)
        self._w2 = self._weights.get(W.moe_w2, None)
        self._w1_scale = self._weights.get(W.moe_s1, None)
        self._w2_scale = self._weights.get(W.moe_s2, None)
        assert self._w1 is not None and self._w2 is not None
        # check fp8 block quantization
        if self.quant_config.is_quantized:
            if (
                self.quant_config.quant_dtype == torch.float8_e4m3fn
                and self.quant_config.is_block_quantized
            ):
                if self.quant_config.block_shape != self.DEEPGEMM_BLOCK_SHAPE:
                    raise NotImplementedError(
                        "DeepGemmMaskedExecutor only supports fp8 block quantization with block shape 128x128"
                    )
                self._use_fp8 = True
                assert self._w1_scale is not None and self._w2_scale is not None
            else:
                raise NotImplementedError(
                    "DeepGemmMaskedExecutor only supports fp8 block quantization or bf16"
                )
        else:
            self._use_fp8 = False

        if self._use_fp8 and is_deep_gemm_e8m0_used():
            w1_tmp, self._w1_scale = requant_weight_ue8m0(self._w1, self._w1_scale)
            self._w1.copy_(w1_tmp)
            self._weights[W.moe_s1] = self._w1_scale
            del w1_tmp
            w2_tmp, self._w2_scale = requant_weight_ue8m0(self._w2, self._w2_scale)
            self._w2.copy_(w2_tmp)
            self._weights[W.moe_s2] = self._w2_scale
            del w2_tmp

    @property
    def local_num_experts(self) -> int:
        assert self._w1 is not None
        return self._w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:

        assert self._w1 is not None and self._w2 is not None
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None

        expert_x = payload.expert_x
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        assert expert_num_tokens is not None

        assert expert_x.ndim == 3
        E, M, K = expert_x.size()

        _, N, _ = self._w1.size()
        assert N % 2 == 0
        assert self._w1.size(0) == E
        assert self._w1.size(2) == K
        assert self._w2.size(0) == E
        assert self._w2.size(1) == K
        assert self._w2.size(2) == N // 2

        workspace = torch.empty((E, M, N), device=expert_x.device, dtype=torch.bfloat16)
        output = torch.empty((E, M, K), device=expert_x.device, dtype=torch.bfloat16)

        if self._use_fp8 and not is_deep_gemm_e8m0_used():
            assert self._w1_scale is not None and self._w2_scale is not None
            assert payload.expert_x_scale is not None

            expert_x_scale = payload.expert_x_scale

            assert expert_x_scale.size(0) == E
            assert expert_x_scale.size(1) == M
            assert expert_x_scale.size(2) == K // self.DEEPGEMM_BLOCK_SHAPE[1]

            m_grouped_fp8_gemm_nt_masked(
                (expert_x, expert_x_scale),
                (self._w1, self._w1_scale),
                workspace,
                expert_num_tokens,
                M,
            )
            a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_masked(
                workspace,
                expert_num_tokens,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[1],
                use_ue8m0=is_deep_gemm_e8m0_used(),
                eps=1e-10,
            )
            m_grouped_fp8_gemm_nt_masked(
                (a2q, a2q_scale),
                (self._w2, self._w2_scale),
                output,
                expert_num_tokens,
                M,
            )
        elif self._use_fp8 and is_deep_gemm_e8m0_used():
            assert self._w1_scale is not None and self._w2_scale is not None
            assert payload.expert_x_scale is not None

            expert_x_scale = payload.expert_x_scale
            assert expert_x_scale.size(0) == E
            assert expert_x_scale.size(1) == M
            # assert expert_x_scale.size(2) == K // self.DEEPGEMM_BLOCK_SHAPE[1]
            m_grouped_fp8_gemm_nt_masked(
                (expert_x, expert_x_scale),
                (self._w1, self._w1_scale),
                workspace,
                expert_num_tokens,
                M,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )
            # 老的和新的精度对不齐，新的精度感觉不对，先用老的吧
            down_input, down_input_scale = silu_mul_fp8_quant_deep_gemm_masked(
                workspace,
                expert_num_tokens,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[1],
                use_ue8m0=is_deep_gemm_e8m0_used(),
                eps=1e-10,
            )
            # down_input = torch.empty(
            #     (
            #         workspace.shape[0],
            #         workspace.shape[1],
            #         workspace.shape[2] // 2,
            #     ),
            #     device=workspace.device,
            #     dtype=torch.float8_e4m3fn,
            # )
            # down_input_scale = torch.empty(
            #     (
            #         workspace.shape[0],
            #         workspace.shape[1],
            #         workspace.shape[2] // 2 // self.DEEPGEMM_BLOCK_SHAPE[1],
            #     ),
            #     device=workspace.device,
            #     dtype=torch.float32,
            # )

            # silu_and_mul_masked_post_quant_fwd(
            #     workspace,
            #     down_input,
            #     down_input_scale,
            #     self.DEEPGEMM_BLOCK_SHAPE[1],
            #     expert_num_tokens,
            #     scale_ue8m0=is_deep_gemm_e8m0_used(),
            # )
            # down_input, down_input_scale = sgl_per_token_group_quant_fp8(workspace,
            #                                                              group_size=self.DEEPGEMM_BLOCK_SHAPE[1],
            #                                                              column_major_scales=True,
            #                                                              scale_tma_aligned=True,
            #                                                              scale_ue8m0=is_deep_gemm_e8m0_used(),
            #                                                              fuse_silu_and_mul=True,
            #                                                              masked_m=expert_num_tokens)
            m_grouped_fp8_gemm_nt_masked(
                (down_input, down_input_scale),
                (self._w2, self._w2_scale),
                output,
                expert_num_tokens,
                M,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )
        else:
            m_grouped_bf16_gemm_nt_masked(
                expert_x, self._w1, workspace, expert_num_tokens, M
            )
            a2q = silu_mul_bf16_deep_gemm_masked(
                workspace, expert_num_tokens, group_size=256
            )
            m_grouped_bf16_gemm_nt_masked(a2q, self._w2, output, expert_num_tokens, M)

        return output
