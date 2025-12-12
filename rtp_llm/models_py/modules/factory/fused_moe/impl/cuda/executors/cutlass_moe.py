from math import prod
from typing import Any, Callable, Optional

import torch

import rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe as mm
import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    cutlass_moe_mm_fp8_scaled,
    get_best_config_swap_ab,
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

from .util import moe_kernel_quantize_input, resize_cache


class CutlassExpertsFp8(mm.FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_FP8

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if CutlassExpertsFp8 can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        )

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        num_experts: int,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__(
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=per_act_token_quant,
                per_out_ch_quant=per_out_ch_quant,
                block_shape=block_shape,
            )
        )
        self.w1 = w1
        self.w2 = w2
        self.w1_scale = w1_scale
        self.w2_scale = w2_scale
        self.a1q_scale = a1q_scale
        self.a2_scale = a2_scale
        self.num_experts = num_experts
        assert per_out_ch_quant is False
        assert block_shape is None

        _, K, N = self.w2.shape
        device = self.w2.device
        self.ab_strides1 = torch.full(
            (w1.size(0),), K, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (w1.size(0),), 2 * N, device=device, dtype=torch.int64
        )
        self.ab_strides2 = torch.full(
            (w1.size(0),), N, device=device, dtype=torch.int64
        )
        self.c_strides2 = torch.full((w1.size(0),), K, device=device, dtype=torch.int64)

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        per_act_token = self.quant_config.is_per_act_token
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        expert_num_tokens = (
            payload.expert_tokens_meta.expert_num_tokens if expert_map is None else None
        )
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            num_gemm_tokens = sum(payload.expert_tokens_meta.expert_num_tokens_cpu)
        elif payload.expert_tokens_meta.expert_num_tokens is not None:
            expert_num_tokens_cpu = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
            num_gemm_tokens = sum(expert_num_tokens_cpu)
        else:
            num_gemm_tokens = topk_ids.numel()

        if num_gemm_tokens <= 0:
            return torch.zeros(
                payload.expert_x.shape,
                device=payload.expert_x.device,
                dtype=payload.expert_x_origin_dtype,
            )

        E, _, _ = self.w1.size()
        _, K, N = self.w2.size()
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

        local_topk_ids = torch.where(topk_ids != -1, topk_ids, E)

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
            [num_gemm_tokens, max(N * 2, K)],
            device=payload.expert_x.device,
            dtype=payload.expert_x_origin_dtype,
        )
        workspace2 = torch.empty(
            [max(M, num_gemm_tokens), max(N, K)],
            device=payload.expert_x.device,
            dtype=payload.expert_x_origin_dtype,
        )

        a1q_permute = resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (num_gemm_tokens, K)
        )
        c1 = resize_cache(workspace13, (num_gemm_tokens, N * 2))
        c2 = resize_cache(workspace2, (num_gemm_tokens, N))
        c3 = resize_cache(workspace13, (num_gemm_tokens, K))
        output = resize_cache(workspace2, (M, K))

        swap_ab_gemm1 = get_best_config_swap_ab(E, num_gemm_tokens, 2 * N, K)
        swap_ab_gemm2 = get_best_config_swap_ab(E, num_gemm_tokens, K, N)

        a1q_scale_permute = torch.empty(
            (num_gemm_tokens,), dtype=torch.float32, device=payload.expert_x.device
        )

        problem_sizes1 = torch.empty(
            (E, 3), dtype=torch.int32, device=payload.expert_x.device
        )
        problem_sizes2 = torch.empty(
            (E, 3), dtype=torch.int32, device=payload.expert_x.device
        )
        expert_offsets = torch.empty(
            (E,), dtype=torch.int32, device=payload.expert_x.device
        )
        src_2_dst = cutlass_moe_pre_reorder(
            input=expert_x,
            permuted_input=a1q_permute,
            input_scale=expert_x_scale,
            permuted_scale=a1q_scale_permute,
            topk_ids=local_topk_ids,
            num_local_experts=E,
            topk=topk,
            num_tokens=M,
            hidden_size=K,
        )
        compute_ops.get_cutlass_moe_mm_without_permute_info(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            E,
            N,
            K,
            swap_ab_gemm1,
            swap_ab_gemm2,
        )
        if not per_act_token and expert_map is not None:
            c1.fill_(0)

        cutlass_moe_mm_fp8_scaled(
            c1,
            a1q_permute,
            self.w1,
            a1q_scale_permute,
            self.w1_scale,
            expert_offsets,
            problem_sizes1,
            self.ab_strides1,
            self.ab_strides1,
            self.c_strides1,
            per_act_token,
            False,
            num_gemm_tokens,
            swap_ab_gemm1,
        )

        silu_and_mul(c2, c1)
        a2q, a2q_scale = moe_kernel_quantize_input(
            c2, a2_scale, torch.float8_e4m3fn, per_act_token
        )

        if expert_map is not None:
            c3.fill_(0)

        cutlass_moe_mm_fp8_scaled(
            c3,
            a2q,
            self.w2,
            a2q_scale,
            self.w2_scale,
            expert_offsets,
            problem_sizes2,
            self.ab_strides2,
            self.ab_strides2,
            self.c_strides2,
            per_act_token,
            False,
            num_gemm_tokens,
            swap_ab_gemm2,
        )
        del a2q

        post_reorder_triton_kernel[(M,)](
            down_output_ptr=c3,
            output_ptr=output,
            src2dst_ptr=src_2_dst,
            topk_ids_ptr=topk_ids,
            topk_weights_ptr=topk_weights,
            topk=topk,
            hidden_size=K,
            BLOCK_SIZE=512,
        )
        return output


class CutlassBatchedExpertsFp8(mm.FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_BATCHED_FP8

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if CutlassBatchedExpertsFp8 can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        )

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        num_experts: int,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__(
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=per_act_token_quant,
                per_out_ch_quant=per_out_ch_quant,
                block_shape=block_shape,
            )
        )
        self.w1 = w1
        self.w2 = w2
        self.w1_scale = w1_scale
        self.w2_scale = w2_scale
        self.a1q_scale = a1q_scale
        self.a2_scale = a2_scale

        self.num_local_experts = self.w1.size(0)
        self.num_experts = num_experts

        assert per_out_ch_quant is False
        assert block_shape is None
        _, K, N = self.w2.shape
        device = self.w2.device
        self.ab_strides1 = torch.full(
            (w1.size(0),), K, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (w1.size(0),), 2 * N, device=device, dtype=torch.int64
        )
        self.ab_strides2 = torch.full(
            (w1.size(0),), N, device=device, dtype=torch.int64
        )
        self.c_strides2 = torch.full((w1.size(0),), K, device=device, dtype=torch.int64)

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        topk_ids = payload.expert_topk_ids
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        E, _, _ = self.w1.size()
        _, K, N = self.w2.shape
        M = payload.expert_x.size(1)
        topk = topk_ids.size(0)
        assert payload.expert_x.dim() == 3
        assert payload.expert_x.size(0) == E
        assert topk_ids.dim() == 2
        assert activation == "SiGLU"

        per_act_token = self.quant_config.is_per_act_token

        if payload.expert_x.dtype is not torch.float8_e4m3fn:
            assert payload.expert_x.dtype == torch.bfloat16
            assert payload.expert_x_scale is None
            # per tensor quant bf16 input to fp8
            if not per_act_token:
                E, M, H = payload.expert_x.shape
                x = payload.expert_x.view(-1, H)
                if torch.sum(expert_num_tokens) > 0:
                    # TODO(serina.wzq): use high performance kernel impl
                    index = torch.arange(
                        M,
                        dtype=expert_num_tokens.dtype,
                        device=expert_num_tokens.device,
                    ).repeat(E, 1)
                    input_mask = (index < (expert_num_tokens.view(-1, 1))).view(-1)
                    scale_inv = (
                        x[input_mask].abs().max() / torch.finfo(torch.float8_e4m3fn).max
                    )
                    scale = torch.tensor(
                        [scale_inv], dtype=torch.float32, device=x.device
                    )
                else:
                    scale = torch.tensor([1], dtype=torch.float32, device=x.device)
                q_x, expert_x_scale = moe_kernel_quantize_input(
                    x, scale, torch.float8_e4m3fn, False, None
                )
                expert_x = q_x.view(E, -1, H)
        else:
            assert payload.expert_x_scale is not None
            expert_x = payload.expert_x
            expert_x_scale = payload.expert_x_scale

        workspace_dtype = payload.expert_x_origin_dtype
        workspace13 = torch.empty(
            [self.local_num_experts, M, max(2 * N, K)],
            device=expert_x.device,
            dtype=workspace_dtype,
        )
        workspace2 = torch.empty(
            [self.local_num_experts, M, N],
            device=expert_x.device,
            dtype=workspace_dtype,
        )
        output = torch.empty(
            [self.local_num_experts, M, K],
            device=expert_x.device,
            dtype=workspace_dtype,
        )

        elements_m = topk_ids.numel() * self.local_num_experts // self.num_experts
        swap_ab_gemm1 = get_best_config_swap_ab(
            self.local_num_experts, elements_m, 2 * N, K
        )
        swap_ab_gemm2 = get_best_config_swap_ab(
            self.local_num_experts, elements_m, K, N
        )

        c1 = resize_cache(workspace13, (self.local_num_experts * M, N * 2))
        c2 = resize_cache(workspace2, (self.local_num_experts * M, N))
        c3 = resize_cache(workspace13, (self.local_num_experts * M, K))
        expert_offsets = torch.empty(
            (self.local_num_experts), dtype=torch.int32, device=expert_x.device
        )
        problem_sizes1 = torch.empty(
            (self.local_num_experts, 3), dtype=torch.int32, device=expert_x.device
        )
        problem_sizes2 = torch.empty(
            (self.local_num_experts, 3), dtype=torch.int32, device=expert_x.device
        )

        compute_ops.get_cutlass_batched_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            self.local_num_experts,
            M,
            N,
            K,
            swap_ab_gemm1,
            swap_ab_gemm2,
        )

        w1_scale = self.w1_scale.reshape(self.w1_scale.size(0), -1)
        w2_scale = self.w2_scale.reshape(self.w2_scale.size(0), -1)
        expert_x = expert_x.reshape(-1, expert_x.size(2))
        expert_x_scale = (
            expert_x_scale
            if not per_act_token
            else expert_x_scale.reshape(-1, expert_x_scale.size(2)).contiguous()
        )
        if not per_act_token:
            c1.fill_(0)

        cutlass_moe_mm_fp8_scaled(
            c1,
            expert_x,
            self.w1,
            expert_x_scale,
            w1_scale,
            expert_offsets,
            problem_sizes1,
            self.ab_strides1,
            self.ab_strides1,
            self.c_strides1,
            per_act_token,
            False,
            elements_m,
            swap_ab_gemm1,
        )

        a2q, a2q_scale = silu_mul_fp8_per_token_quant_batched(c1, expert_num_tokens)

        if expert_map is not None:
            output.fill_(0)

        cutlass_moe_mm_fp8_scaled(
            output.reshape(-1, K),
            a2q,
            self.w2,
            a2q_scale,
            w2_scale,
            expert_offsets,
            problem_sizes2,
            self.ab_strides2,
            self.ab_strides2,
            self.c_strides2,
            per_act_token,
            False,
            elements_m,
            swap_ab_gemm2,
        )
        return output
