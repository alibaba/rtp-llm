from typing import Any, Dict, Optional

import torch

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.models_py.kernels.cuda.w4a8_kernel import (
    w4a8_group_gemm_ptpc,
)
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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
    silu_mul_fp8_per_token_quant_batched,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    cutlass_moe_pre_reorder,
    post_reorder_triton_kernel,
)
from rtp_llm.utils.model_weight import W

from .util import moe_kernel_quantize_input, resize_cache


class CutlassExpertsW4a8Int4(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_W4A8_INT4

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if CutlassExpertsW4a8Int4 can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["W4A8_INT4_PER_CHANNEL"]
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        # Update quant_config with W4A8-INT4-specific settings
        self.quant_config.quant_dtype = torch.float8_e4m3fn
        self.quant_config.per_act_token_quant = True
        self.quant_config.per_out_ch_quant = False
        self.quant_config.block_shape = None

        self.num_experts = config.expert_num

        self.swap_ab_gemm1 = True
        self.swap_ab_gemm2 = True

        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]
        self.a1q_scale = weights.get(W.moe_w1_input_sr, None)
        self.a2_scale = weights.get(W.moe_w2_input_sr, None)

        # Setup strides for cutlass kernel
        self.E = self.w2.shape[0]
        self.K = self.w2.shape[1]
        self.N = self.w2.shape[2] * 2
        scale_k = self.w1_scale.shape[2]
        assert (self.K % scale_k == 0), f"invalid params {self.K} or {scale_k}"
        self.group_size = self.K // scale_k
        device = self.w2.device
        self.ab_strides1 = torch.full((self.E,), self.K, device=device, dtype=torch.int64)
        self.b_scales_strides1 = torch.tensor(
            [2 * self.N, 0], dtype=torch.int64, device=device).unsqueeze(0).repeat(self.E, 1, 1)
        self.c_strides1 = torch.full((self.E,), 2 * self.N, device=device, dtype=torch.int64)
        self.ab_strides2 = torch.full((self.E,), self.N, device=device, dtype=torch.int64)
        self.b_scales_strides2 = torch.tensor(
            [self.K, 0], dtype=torch.int64, device=device).unsqueeze(0).repeat(self.E, 1, 1)
        self.c_strides2 = torch.full((self.E,), self.K, device=device, dtype=torch.int64)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        per_act_token = self.quant_config.is_per_act_token
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        if (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens_cpu is not None
        ):
            assert isinstance(
                payload.expert_tokens_meta.expert_num_tokens_cpu, list
            ), "expert_num_tokens_cpu should be a list"
            num_gemm_tokens = sum(payload.expert_tokens_meta.expert_num_tokens_cpu)
        elif (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        ):
            expert_num_tokens_cpu = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
            num_gemm_tokens = sum(expert_num_tokens_cpu)
        else:
            num_gemm_tokens = topk_ids.numel()

        if num_gemm_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    payload.expert_x.shape,
                    device=payload.expert_x.device,
                    dtype=payload.expert_x_origin_dtype,
                ),
            )

        assert payload.expert_x.dim() == 2
        assert topk_ids.size(0) == payload.expert_x.size(0)
        assert topk_ids.dim() == 2
        assert activation == "SiGLU"
        M = payload.expert_x.size(0)
        topk = topk_ids.size(1)

        if expert_map is not None:
            topk_ids = torch.where(expert_map[topk_ids] != -1, expert_map[topk_ids], -1)
        else:
            topk_ids = topk_ids.to(torch.int32)

        local_topk_ids = torch.where(topk_ids != -1, topk_ids, self.E)

        if payload.expert_x.dtype is not torch.float8_e4m3fn:
            assert payload.expert_x.dtype == torch.bfloat16
            assert payload.expert_x_scale is None
            # quant bf16 inputs to fp8
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                payload.expert_x,
                None,
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=per_act_token,
                block_shape=None,
            )
        else:
            assert payload.expert_x_scale is not None
            expert_x = payload.expert_x
            expert_x_scale = payload.expert_x_scale

        workspace13 = torch.empty(
            [num_gemm_tokens, max(self.N * 2, self.K)],
            device=payload.expert_x.device,
            dtype=payload.expert_x_origin_dtype,
        )
        workspace2 = torch.empty(
            [max(M, num_gemm_tokens), max(self.N, self.K)],
            device=payload.expert_x.device,
            dtype=payload.expert_x_origin_dtype,
        )

        a1q_permute = resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (num_gemm_tokens, self.K)
        )
        c1 = resize_cache(workspace13, (num_gemm_tokens, self.N * 2))
        c2 = resize_cache(workspace2, (num_gemm_tokens, self.N))
        c3 = resize_cache(workspace13, (num_gemm_tokens, self.K))
        output = resize_cache(workspace2, (M, self.K))

        a1q_scale_permute = torch.empty(
            (num_gemm_tokens,), dtype=torch.float32, device=payload.expert_x.device
        )

        problem_sizes1 = torch.empty(
            (self.E, 3), dtype=torch.int32, device=payload.expert_x.device
        )
        problem_sizes2 = torch.empty(
            (self.E, 3), dtype=torch.int32, device=payload.expert_x.device
        )
        expert_offsets = torch.empty(
            (self.E,), dtype=torch.int32, device=payload.expert_x.device
        )
        src_2_dst = cutlass_moe_pre_reorder(
            input=expert_x,
            permuted_input=a1q_permute,
            input_scale=expert_x_scale,
            permuted_scale=a1q_scale_permute,
            topk_ids=local_topk_ids,
            num_local_experts=self.E,
            topk=topk,
            num_tokens=M,
            hidden_size=self.K,
        )
        compute_ops.get_cutlass_moe_mm_without_permute_info(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            self.E,
            self.N,
            self.K,
            self.swap_ab_gemm1,
            self.swap_ab_gemm2,
        )
        if not per_act_token and expert_map is not None:
            c1.fill_(0)

        w4a8_group_gemm_ptpc(
            c1,
            a1q_permute,
            self.w1,
            self.w1_scale,
            a1q_scale_permute,
            expert_offsets,
            problem_sizes1,
            self.ab_strides1,
            self.ab_strides1,
            self.b_scales_strides1,
            self.c_strides1,
            self.group_size
        )

        silu_and_mul(c2, c1)
        a2q, a2q_scale = moe_kernel_quantize_input(
            c2, a2_scale, torch.float8_e4m3fn, per_act_token
        )

        if expert_map is not None:
            c3.fill_(0)

        w4a8_group_gemm_ptpc(
            c3,
            a2q,
            self.w2,
            self.w2_scale,
            a2q_scale,
            expert_offsets,
            problem_sizes2,
            self.ab_strides2,
            self.ab_strides2,
            self.b_scales_strides2,
            self.c_strides2,
            self.group_size
        )
        del a2q

        post_reorder_triton_kernel[(M,)](
            down_output_ptr=c3,
            output_ptr=output,
            src2dst_ptr=src_2_dst,
            topk_ids_ptr=topk_ids,
            topk_weights_ptr=topk_weights,
            topk=topk,
            hidden_size=self.K,
            BLOCK_SIZE=512,
        )
        return CombineForwardPayload(fused_expert_output=output)


class CutlassBatchedExpertsW4a8Int4(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_BATCHED_W4A8_INT4

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if CutlassBatchedExpertsW4a8Int4 can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["W4A8_INT4_PER_CHANNEL"]
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        # Update quant_config with W4A8-INT4-specific settings
        self.quant_config.quant_dtype = torch.float8_e4m3fn
        self.quant_config.per_act_token_quant = True
        self.quant_config.per_out_ch_quant = False
        self.quant_config.block_shape = None

        self.num_experts = config.expert_num

        self.swap_ab_gemm1 = True
        self.swap_ab_gemm2 = True

        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]
        self.a1q_scale = weights.get(W.moe_w1_input_sr, None)
        self.a2_scale = weights.get(W.moe_w2_input_sr, None)

        # Setup strides for cutlass kernel
        self.E = self.w2.shape[0]
        self.K = self.w2.shape[1]
        self.N = self.w2.shape[2] * 2
        scale_k = self.w1_scale.shape[2]
        assert (self.K % scale_k == 0), f"invalid params {self.K} or {scale_k}"
        self.group_size = self.K // scale_k
        device = self.w2.device
        self.ab_strides1 = torch.full((self.E,), self.K, device=device, dtype=torch.int64)
        self.b_scales_strides1 = torch.tensor(
            [2 * self.N, 0], dtype=torch.int64, device=device).unsqueeze(0).repeat(self.E, 1, 1)
        self.c_strides1 = torch.full((self.E,), 2 * self.N, device=device, dtype=torch.int64)
        self.ab_strides2 = torch.full((self.E,), self.N, device=device, dtype=torch.int64)
        self.b_scales_strides2 = torch.tensor(
            [self.K, 0], dtype=torch.int64, device=device).unsqueeze(0).repeat(self.E, 1, 1)
        self.c_strides2 = torch.full((self.E,), self.K, device=device, dtype=torch.int64)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_tokens_meta is not None
        assert payload.expert_topk_ids is not None
        topk_ids = payload.expert_topk_ids
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        M = payload.expert_x.size(1)
        assert payload.expert_x.dim() == 3
        assert payload.expert_x.size(0) == self.E
        assert topk_ids.dim() == 2
        assert activation == "SiGLU"

        assert payload.expert_x.dtype == torch.float8_e4m3fn
        assert payload.expert_x_scale is not None
        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale

        workspace_dtype = payload.expert_x_origin_dtype
        workspace13 = torch.empty(
            [self.E, M, max(2 * self.N, self.K)],
            device=expert_x.device,
            dtype=workspace_dtype,
        )
        output = torch.empty(
            [self.E, M, self.K],
            device=expert_x.device,
            dtype=workspace_dtype,
        )

        c1 = resize_cache(workspace13, (self.E * M, self.N * 2))
        expert_offsets = torch.empty(
            (self.E), dtype=torch.int32, device=expert_x.device
        )
        problem_sizes1 = torch.empty(
            (self.E, 3), dtype=torch.int32, device=expert_x.device
        )
        problem_sizes2 = torch.empty(
            (self.E, 3), dtype=torch.int32, device=expert_x.device
        )

        compute_ops.get_cutlass_batched_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            self.E,
            M,
            self.N,
            self.K,
            self.swap_ab_gemm1,
            self.swap_ab_gemm2
        )

        expert_x = expert_x.reshape(-1, expert_x.size(2))
        expert_x_scale = expert_x_scale.reshape(-1, expert_x_scale.size(2)).contiguous()

        w4a8_group_gemm_ptpc(
            c1,
            expert_x,
            self.w1,
            self.w1_scale,
            expert_x_scale,
            expert_offsets,
            problem_sizes1,
            self.ab_strides1,
            self.ab_strides1,
            self.b_scales_strides1,
            self.c_strides1,
            self.group_size
        )

        a2q, a2q_scale = silu_mul_fp8_per_token_quant_batched(c1, expert_num_tokens)

        if expert_map is not None:
            output.fill_(0)

        w4a8_group_gemm_ptpc(
            output.reshape(-1, self.K),
            a2q,
            self.w2,
            self.w2_scale,
            a2q_scale,
            expert_offsets,
            problem_sizes2,
            self.ab_strides2,
            self.ab_strides2,
            self.b_scales_strides2,
            self.c_strides2,
            self.group_size
        )
        return CombineForwardPayload(fused_expert_output=output)
