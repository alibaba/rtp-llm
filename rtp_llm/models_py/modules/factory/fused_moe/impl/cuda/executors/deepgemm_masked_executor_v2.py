import math
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
    silu_and_mul_masked_post_quant_packed_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    tma_align_input_scale,
)
from rtp_llm.models_py.utils.math import ceil_div, align
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


class DeepGemmMaskedExecutorV2(FusedMoeExpertExecutor):
    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmMaskedExecutorV2 can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num

        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.activation = config.activation_type
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True

        self.max_moe_normal_masked_token_num = config.max_moe_normal_masked_token_num

        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1]
        self.w2_weight_scale_inv = weights[W.moe_s2]
        self.w13_weight_scale = None
        self.w2_weight_scale = None

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        if is_deep_gemm_e8m0_used():
            w13_weight_tmp, self.w13_weight_scale_inv = requant_weight_ue8m0(
                self.w13_weight, self.w13_weight_scale_inv
            )
            self.w13_weight.copy_(w13_weight_tmp)
            weights[W.moe_s1] = self.w13_weight_scale_inv
            del w13_weight_tmp
            w2_weight_tmp, self.w2_weight_scale_inv = requant_weight_ue8m0(
                self.w2_weight, self.w2_weight_scale_inv
            )
            self.w2_weight.copy_(w2_weight_tmp)
            weights[W.moe_s2] = self.w2_weight_scale_inv
            del w2_weight_tmp

        self.w13_weight_fp8 = (
            self.w13_weight,
            self.w13_weight_scale_inv,
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

        self.num_gemm_sms = get_num_device_sms()

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        token_num = payload.expert_x.shape[0]
        if token_num <= self.max_moe_normal_masked_token_num:
            return self.execute_masked(payload, activation, expert_map, a2_scale, apply_router_weight_on_input, extra_expert_args)
        else:
            return self.execute_contiguous(payload, activation, expert_map, a2_scale, apply_router_weight_on_input, extra_expert_args)

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        assert payload.expert_tokens_meta.expert_num_tokens is not None

        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            hidden_states_fp8 = payload.expert_x
            hidden_states_scale = payload.expert_x_scale
            topk_idx = payload.expert_topk_ids
            topk_weights = payload.expert_topk_weights
            num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

            token_num = hidden_states_fp8.shape[0]
            num_experts = num_recv_tokens_per_expert.shape[0]
            max_token_num = token_num * self.top_k
            token_num_mean_per_expert = ceil_div(max_token_num, num_experts)
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, token_num_mean_per_expert)

            _, K = hidden_states_fp8.size()
            assert K == self.K, f"K: {K} != self.K: {self.K}"
            hidden_states_fp8_shape = hidden_states_fp8.shape
            hidden_states_fp8_device = hidden_states_fp8.device
            input_tensor = [
                torch.empty(
                    (self.num_experts_per_partition, alignment, self.K),
                    device=hidden_states_fp8_device,
                    dtype=hidden_states_fp8.dtype,
                ),
                (
                    torch.zeros(
                        [ceil_div(self.K // self.BLOCK_SIZE, 4), self.num_experts_per_partition, alignment],
                        device=hidden_states_fp8_device,
                        dtype=torch.int,
                    ).permute(1, 2, 0)
                    if is_deep_gemm_e8m0_used()
                    else torch.empty(
                        (self.num_experts_per_partition, alignment, self.K // self.BLOCK_SIZE),
                        device=hidden_states_fp8_device,
                        dtype=torch.float32,
                    )
                ),
            ]
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)
            ep_scatter_v2(
                hidden_states_fp8,
                hidden_states_scale,
                topk_idx,
                alignment,
                expert_start_loc,
                input_tensor[0].view(self.num_experts_per_partition * alignment, self.K),
                input_tensor[1].view(self.num_experts_per_partition * alignment, -1),
                output_index,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
            dispose_tensor(hidden_states_fp8)
            if not is_deep_gemm_e8m0_used():
                input_tensor[1] = tma_align_input_scale(input_tensor[1])

            upgate_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.N),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            # Gate and Up GroupGEMM-0
            m_grouped_fp8_gemm_nt_masked(
                (input_tensor[0], input_tensor[1]),
                self.w13_weight_fp8,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            del input_tensor
            # Allocate down_input
            down_input = torch.empty(
                (self.num_experts_per_partition, alignment, self.N // 2),
                device=hidden_states_fp8_device,
                dtype=torch.float8_e4m3fn,
            )

            # SM100 (compute capability 10.x) uses fused packed kernel for better performance
            # when UE8M0 scale format is enabled
            sm_major = torch.cuda.get_device_capability()[0]
            if (
                sm_major == 10
                and is_deep_gemm_e8m0_used()
                and self.N % (self.DEEPGEMM_BLOCK_SHAPE[0] * 2 * 4) == 0
            ):
                # Create packed scale tensor with proper layout for deep_gemm
                # Shape: (E, T, G // 4) where G = hidden_dim // 2 // group_size
                down_input_scale = create_packed_scale_tensor(
                    expert_num=self.num_experts_per_partition,
                    token_num_padded=alignment,
                    hidden_dim=self.N,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    device=hidden_states_fp8_device,
                )
                # Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing
                silu_and_mul_masked_post_quant_packed_fwd(
                    upgate_output,
                    down_input,
                    down_input_scale,
                    self.DEEPGEMM_BLOCK_SHAPE[0],
                    num_recv_tokens_per_expert,
                )
            else:
                # Standard path for other SM versions
                down_input_scale = torch.empty(
                    (
                        self.num_experts_per_partition,
                        alignment,
                        self.N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0],
                    ),
                    device=hidden_states_fp8_device,
                    dtype=torch.float32,
                )
                # SiLU Activation
                silu_mul_masked_fp8_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=num_recv_tokens_per_expert,
                    expected_m=expected_m,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )

            # Free upgate_output
            dispose_tensor(upgate_output)
            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.K),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )

            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                (
                    down_input,
                    down_input_scale,
                ),
                self.w2_weight_fp8,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)

            gather_out = torch.empty(
                hidden_states_fp8_shape,
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            ep_gather(down_output.view(self.num_experts_per_partition * alignment,
                      self.K), topk_idx, topk_weights, output_index, gather_out)
            return CombineForwardPayload(fused_expert_output=gather_out)

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_idx = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            num_recv_tokens_per_expert = (
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
        elif payload.expert_tokens_meta.expert_num_tokens is not None:
            num_recv_tokens_per_expert = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
        else:
            raise ValueError(
                "expert_tokens_meta.expert_num_tokens or expert_tokens_meta.expert_num_tokens_cpu should be not None"
            )
        if isinstance(num_recv_tokens_per_expert, torch.Tensor):
            num_recv_tokens_per_expert = num_recv_tokens_per_expert.tolist()
        num_recv_tokens_per_expert = [
            align_up_math(x, self.EXPERT_ALIGNMENT) for x in num_recv_tokens_per_expert
        ]
        all_tokens: int = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                ),
            )
        _, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            (
                torch.zeros(
                    [ceil_div(K // self.BLOCK_SIZE, 4), all_tokens],
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if is_deep_gemm_e8m0_used()
                else torch.empty(
                    (all_tokens, K // self.BLOCK_SIZE),
                    device=hidden_states_fp8.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=is_deep_gemm_e8m0_used(),
        )
        dispose_tensor(hidden_states_fp8)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not is_deep_gemm_e8m0_used():
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        m_grouped_fp8_gemm_nt_contiguous(
            (input_tensor[0], input_tensor[1]),
            self.w13_weight_fp8,
            gateup_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        gateup_output = gateup_output.view(-1, N)
        silu_and_mul(down_input, gateup_output)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if is_deep_gemm_e8m0_used():
            down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
                down_input,
                group_size=self.BLOCK_SIZE,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
        else:
            down_input_fp8, down_input_scale = trt_fp8_quantize_128(down_input, False)
        del down_input
        if not is_deep_gemm_e8m0_used():
            down_input_scale = tma_align_input_scale(down_input_scale)
        m_grouped_fp8_gemm_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        del down_input_fp8, down_input_scale
        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        return CombineForwardPayload(fused_expert_output=gather_out)
