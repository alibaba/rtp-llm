# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
import logging
import math
import os
from contextlib import nullcontext
from functools import cache
from typing import Any, Dict, Optional

import torch
import triton.language as tl

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_mk_alignment,
    configure_deep_gemm_num_sms,
    get_theoretical_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
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
    create_packed_scale_tensor,
    silu_and_mul,
    silu_and_mul_masked_post_quant_packed_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    tma_align_input_scale,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config as get_triton_moe_config,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel as invoke_triton_moe_kernel,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    moe_align_block_size_compiled,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W

_SM120_TRITON_MIN_TOKENS = 1
_SM120_TRITON_MAX_TOKENS = 32
_SM120_TRITON_MAX_TOKENS_ENV = "RTP_LLM_SM120_TRITON_FP8_MAX_TOKENS"
_CUDA_GRAPH_WARMUP_FORWARD_ENV = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"
_SM120_TUNED_FP8_CONFIGS = {
    # Qwen3-30B-A3B, TP=1. Exhaustive search over BM={8,16,32},
    # BN={64,128,256}, BK={64,128}, warps={4,8}, and stages={2,3,4,5}.
    (128, 1536, 2048, 8): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    (128, 2048, 768, 8): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    },
}


def _get_sm120_triton_max_tokens() -> int:
    raw_max_tokens = os.environ.get(
        _SM120_TRITON_MAX_TOKENS_ENV,
        str(_SM120_TRITON_MAX_TOKENS),
    )
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError as error:
        raise ValueError(
            f"{_SM120_TRITON_MAX_TOKENS_ENV} must be an integer, "
            f"got {raw_max_tokens!r}"
        ) from error
    if not 0 <= max_tokens <= _SM120_TRITON_MAX_TOKENS:
        raise ValueError(
            f"{_SM120_TRITON_MAX_TOKENS_ENV} must be in "
            f"[0, {_SM120_TRITON_MAX_TOKENS}], got {max_tokens}"
        )
    return max_tokens


def _is_cuda_graph_warmup_or_capture() -> bool:
    """Return whether this forward is preparing or capturing a CUDA graph.

    ``enable_cuda_graph`` is an engine-level capability and is also true for
    ordinary eager prefill forwards.  The C++ graph runner marks its eager
    warmup with an environment flag, while PyTorch exposes the subsequent
    capture directly.  Requiring either signal keeps short eager prefill out
    of the static ``torch.compile(dynamic=False)`` routing path.
    """
    return (
        os.environ.get(_CUDA_GRAPH_WARMUP_FORWARD_ENV) == "1"
        or torch.cuda.is_current_stream_capturing()
    )


@cache
def _log_sm120_triton_fp8_path(
    min_tokens: int, max_tokens: int, config_source: str
) -> None:
    if max_tokens == 0:
        logger.info(
            "SM120 CUDA Graph Triton FP8 MoE is disabled by %s=0",
            _SM120_TRITON_MAX_TOKENS_ENV,
        )
        return
    logger.info(
        "SM120 CUDA Graph MoE uses Triton FP8 for %d-%d tokens; "
        "config source=%s; other shapes use DeepGEMM (set %s=0 to disable)",
        min_tokens,
        max_tokens,
        config_source,
        _SM120_TRITON_MAX_TOKENS_ENV,
    )


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


def get_sm120_triton_fp8_config(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
) -> Dict[str, Any]:
    """Return SM120 FP8 MoE configs without changing the SM90 defaults."""
    tuned_config = _SM120_TUNED_FP8_CONFIGS.get((E, N, K, top_k))
    if tuned_config is not None and M <= _SM120_TRITON_MAX_TOKENS:
        return dict(tuned_config)

    # Keep the SM120 override pure even if the generic selector starts caching
    # configs in the future.
    config = dict(get_triton_moe_config(M, E, N, K, top_k))
    avg_tokens_per_expert = M * top_k / max(E, 1)
    if avg_tokens_per_expert <= 2:
        # Generic SM120 small-batch fallback from the original optimization.
        config.update(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        )
    return config


class DeepGemmHybridExecutor(FusedMoeExpertExecutor):
    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmHybridExecutor can handle the configuration"""
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
        checker.check(not config.enable_cuda_graph or get_sm()[0] == 12)

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

        self.masked_max_token_num = config.masked_max_token_num
        self.enable_cuda_graph = config.enable_cuda_graph
        self.is_sm120 = get_sm()[0] == 12
        if self.is_sm120 and self.enable_cuda_graph and self.ep_size == 1:
            self.sm120_triton_max_tokens = _get_sm120_triton_max_tokens()
        else:
            # Do not let an SM120-only rollback knob affect H20/SM100 or an
            # executor configuration that can never enter the Triton path.
            self.sm120_triton_max_tokens = 0

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

        if self.is_sm120 and self.enable_cuda_graph and self.ep_size == 1:
            gate_shape = (self.E, self.N, self.K, self.top_k)
            down_shape = (self.E, self.K, self.N // 2, self.top_k)
            config_source = (
                "tuned"
                if gate_shape in _SM120_TUNED_FP8_CONFIGS
                and down_shape in _SM120_TUNED_FP8_CONFIGS
                else "generic"
            )
            _log_sm120_triton_fp8_path(
                _SM120_TRITON_MIN_TOKENS,
                self.sm120_triton_max_tokens,
                config_source,
            )

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
        # This local routed path neither remaps partitioned expert ids nor
        # participates in an EP dispatch/combine collective.  expert_map=None
        # alone does not imply that the payload follows the non-EP contract.
        if (
            self.is_sm120
            and self.enable_cuda_graph
            and self.ep_size == 1
            and _is_cuda_graph_warmup_or_capture()
            and token_num >= _SM120_TRITON_MIN_TOKENS
            and token_num <= self.sm120_triton_max_tokens
            and expert_map is None
            and activation == "SiGLU"
            and not apply_router_weight_on_input
        ):
            return self.execute_triton_fp8(
                payload,
                activation,
                apply_router_weight_on_input,
            )
        # The contiguous path uses a shape-derived fixed workspace and performs
        # all routing metadata work on GPU, so it is safe to capture/replay.
        # It avoids the E * padded_M masked layout that dominates small decode
        # batches on SM120.
        if token_num <= self.masked_max_token_num and not self.enable_cuda_graph:
            return self.execute_masked(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        else:
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )

    def execute_triton_fp8(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> CombineForwardPayload:
        """Run small SM120 decode batches without per-expert 64-row padding."""
        if not self.is_sm120:
            raise RuntimeError("Triton FP8 MoE fast path requires SM120")
        if self.ep_size != 1:
            raise ValueError("Triton FP8 MoE fast path requires ep_size == 1")
        if activation != "SiGLU":
            raise ValueError("Triton FP8 MoE fast path only supports SiGLU")
        if apply_router_weight_on_input:
            raise ValueError(
                "Triton FP8 MoE fast path does not support router weight on input"
            )
        if payload.expert_x is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_x")
        if payload.expert_x_scale is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_x_scale")
        if payload.expert_topk_ids is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_topk_ids")
        if payload.expert_topk_weights is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_topk_weights")

        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        token_num, hidden_size = hidden_states_fp8.shape
        topk = topk_ids.shape[1]
        expert_num, gate_up_size, _ = self.w13_weight.shape
        intermediate_size = gate_up_size // 2

        config1 = get_sm120_triton_fp8_config(
            token_num, expert_num, gate_up_size, hidden_size, topk
        )
        config2 = get_sm120_triton_fp8_config(
            token_num, expert_num, hidden_size, intermediate_size, topk
        )
        block_m = min(config1["BLOCK_SIZE_M"], config2["BLOCK_SIZE_M"])
        config1["BLOCK_SIZE_M"] = block_m
        config2["BLOCK_SIZE_M"] = block_m
        # The execute() gate restricts this path to EP=1, where SelectTopk
        # produces global expert ids in [0, expert_num).  EP padding sentinels
        # must stay on the contiguous path, which owns their remapping logic.
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_compiled(topk_ids, block_m, expert_num)
        )

        route_num = token_num * topk
        gate_up_output = torch.empty(
            (route_num, gate_up_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        invoke_triton_moe_kernel(
            hidden_states_fp8,
            self.w13_weight,
            gate_up_output,
            topk_weights.view(-1),
            topk_ids.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config1,
            tl.bfloat16,
            A_scale=hidden_states_scale,
            B_scale=self.w13_weight_scale_inv,
            block_shape=self.DEEPGEMM_BLOCK_SHAPE,
            scale_ue8m0=True,
        )

        down_input = torch.empty(
            (route_num, intermediate_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(down_input, gate_up_output)
        down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
            down_input,
            group_size=self.BLOCK_SIZE,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        down_output = torch.empty(
            (route_num, hidden_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        invoke_triton_moe_kernel(
            down_input_fp8,
            self.w2_weight,
            down_output,
            topk_weights.view(-1),
            topk_ids.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config2,
            tl.bfloat16,
            A_scale=down_input_scale,
            B_scale=self.w2_weight_scale_inv,
            block_shape=self.DEEPGEMM_BLOCK_SHAPE,
            scale_ue8m0=True,
        )
        output = down_output.view(token_num, topk, hidden_size).sum(dim=1)
        return CombineForwardPayload(fused_expert_output=output)

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
                        [
                            self.num_experts_per_partition,
                            ceil_div(self.K // self.BLOCK_SIZE, 4),
                            alignment,
                        ],
                        device=hidden_states_fp8_device,
                        dtype=torch.int,
                    ).transpose(1, 2)
                    if is_deep_gemm_e8m0_used()
                    else torch.empty(
                        (
                            self.num_experts_per_partition,
                            alignment,
                            self.K // self.BLOCK_SIZE,
                        ),
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
                input_tensor[0].view(
                    self.num_experts_per_partition * alignment, self.K
                ),
                input_tensor[1],
                output_index,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
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
            ep_gather(
                down_output.view(self.num_experts_per_partition * alignment, self.K),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
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
        num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens
        if num_recv_tokens_per_expert is None:
            raise ValueError("expert_num_tokens GPU tensor is required")

        num_experts_local = num_recv_tokens_per_expert.shape[0]
        routed_tokens = hidden_states_fp8.shape[0] * topk_idx.shape[1]
        if routed_tokens == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                )
            )
        if self.is_sm120:
            expert_alignment = min(
                self.EXPERT_ALIGNMENT,
                get_theoretical_mk_alignment_for_contiguous_layout(
                    routed_tokens, num_experts_local
                ),
            )
            max_active_experts = min(routed_tokens, num_experts_local)
            all_tokens = align_up_math(
                routed_tokens + max_active_experts * (expert_alignment - 1),
                expert_alignment,
            )
        else:
            # Preserve the pre-upgrade SM9x/SM100x layout: fixed 128-token
            # expert alignment and CPU-padded metadata.
            expert_alignment = self.EXPERT_ALIGNMENT
            max_active_experts = min(routed_tokens, num_experts_local)
            all_tokens = align_up_math(
                routed_tokens + max_active_experts * (expert_alignment - 1),
                expert_alignment,
            )

        if not self.enable_cuda_graph:
            num_recv_tokens_per_expert_cpu = (
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
            if num_recv_tokens_per_expert_cpu is None:
                num_recv_tokens_per_expert_cpu = (
                    num_recv_tokens_per_expert.cpu().tolist()
                )
            elif isinstance(num_recv_tokens_per_expert_cpu, torch.Tensor):
                num_recv_tokens_per_expert_cpu = num_recv_tokens_per_expert_cpu.tolist()
            actual_aligned = sum(
                align_up_math(int(x), expert_alignment)
                for x in num_recv_tokens_per_expert_cpu
            )
            all_tokens = actual_aligned
            if all_tokens == 0:
                return CombineForwardPayload(
                    fused_expert_output=torch.zeros(
                        hidden_states_fp8.shape,
                        device=hidden_states_fp8.device,
                        dtype=torch.bfloat16,
                    )
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
        m_indices = torch.full(
            (all_tokens,), -1, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.full_like(topk_idx, -1)
        scatter_num_tokens_per_expert = num_recv_tokens_per_expert
        scatter_alignment = expert_alignment
        if not self.is_sm120 and not self.enable_cuda_graph:
            padded_num_tokens_per_expert = [
                align_up_math(int(x), expert_alignment)
                for x in num_recv_tokens_per_expert_cpu
            ]
            scatter_num_tokens_per_expert = torch.tensor(
                padded_num_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
            scatter_alignment = 1
        expert_start_loc = torch.empty_like(scatter_num_tokens_per_expert)
        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            scatter_num_tokens_per_expert,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=is_deep_gemm_e8m0_used(),
            align_m=scatter_alignment,
            derive_counts_from_topk=self.is_sm120 and self.enable_cuda_graph,
        )
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not is_deep_gemm_e8m0_used():
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        if self.is_sm120:
            configure_deep_gemm_mk_alignment_context = configure_deep_gemm_mk_alignment(
                expert_alignment
            )
        else:
            configure_deep_gemm_mk_alignment_context = nullcontext()
        with configure_deep_gemm_mk_alignment_context:
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
        if self.is_sm120:
            configure_deep_gemm_mk_alignment_context = configure_deep_gemm_mk_alignment(
                expert_alignment
            )
        else:
            configure_deep_gemm_mk_alignment_context = nullcontext()
        with configure_deep_gemm_mk_alignment_context:
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
