from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Optional

import torch

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.models_py.modules.moe import TopKWeightAndReduceDelegate
from rtp_llm.models_py.modules.moe.utils import (
    FusedMoEQuantConfig,
    _fp8_perm,
    moe_kernel_quantize_input,
    resize_cache,
)

from libth_transformer.rtp_llm_ops import (  # isort:skip
    cutlass_moe_mm,
    get_cutlass_batched_moe_mm_data,
    get_cutlass_moe_mm_data,
    silu_and_mul,
)


def run_cutlass_moe_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation_callable: Callable,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    a1q_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
):
    a1q = hidden_states
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert a1q.size(-1) == w1.size(2), "Hidden size mismatch w1"
    assert w1.size(1) == w2.size(2) * 2, "Hidden size mismatch w2"
    assert (
        w1_scale.dim() == 1 or w1_scale.size(1) == 1 or w1_scale.shape[1] == w1.size(1)
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1 or w2_scale.size(1) == 1 or w2_scale.shape[1] == w2.size(1)
    ), "W2 scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Expert number mismatch"
    assert (
        a1q_scale is None
        or a1q_scale.dim() == 0
        or a1q_scale.size(0) == 1
        or a1q_scale.size(0) == a1q.shape[0]
    ), "Input scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert w1.size(0) == w1_scale.size(0), "w1 scales expert number mismatch"
    assert w1.size(0) == w2_scale.size(0), "w2 scales expert number mismatch"
    assert (
        a2_scale is None
        or a2_scale.dim() == 0
        or a2_scale.size(0) == 1
        or a2_scale.size(0) == a1q.shape[0]
    ), "Intermediate scale shape mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"
    if expert_map is not None:
        assert expert_num_tokens is None

    M = a1q.size(0)  # non batched expert M
    padded_M = a1q.size(1)  # batched expert M
    _, K, N = w2.shape
    device = a1q.device

    assert w1.size(2) == K
    assert global_num_experts != -1
    assert a1q_scale is not None

    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(
            expert_map[topk_ids] != -1, expert_map[topk_ids], -1
        )
    else:
        local_topk_ids = topk_ids

    topk = local_topk_ids.size(1)
    local_E = w1.size(0)

    if use_batched_format:
        assert expert_num_tokens is not None

        expert_offsets = torch.empty((local_E), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        get_cutlass_batched_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            local_E,
            padded_M,
            N,
            K,
        )
        w1_scale = w1_scale.reshape(w1_scale.size(0), -1)
        w2_scale = w2_scale.reshape(w2_scale.size(0), -1)
        a1q = a1q.reshape(-1, a1q.size(2))
        a1q_scale = (
            a1q_scale
            if not per_act_token
            else a1q_scale.reshape(-1, a1q_scale.size(2)).contiguous()
        )

    else:
        expert_offsets = torch.empty(
            (global_num_experts + 1), dtype=torch.int32, device=device
        )
        problem_sizes1 = torch.empty(
            (global_num_experts, 3), dtype=torch.int32, device=device
        )
        problem_sizes2 = torch.empty(
            (global_num_experts, 3), dtype=torch.int32, device=device
        )

        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        if expert_map is not None:
            a_map = torch.zeros(
                (local_topk_ids.numel()), dtype=torch.int32, device=device
            )
        else:
            a_map = torch.empty(
                (local_topk_ids.numel()), dtype=torch.int32, device=device
            )

        c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)

        get_cutlass_moe_mm_data(
            local_topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            global_num_experts,
            N,
            K,
        )

        a1q = _fp8_perm(a1q, a_map)
        a1q_scale = a1q_scale[a_map] if per_act_token else a1q_scale
        expert_offsets = expert_offsets[:-1]

    ab_strides1 = torch.full((w1.size(0),), K, device=device, dtype=torch.int64)
    c_strides1 = torch.full((w1.size(0),), 2 * N, device=device, dtype=torch.int64)
    ab_strides2 = torch.full((w1.size(0),), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((w1.size(0),), K, device=device, dtype=torch.int64)

    if use_batched_format:
        c1 = resize_cache(workspace13, (local_E * padded_M, N * 2))
        c2 = resize_cache(workspace2, (local_E * padded_M, N))
        c3 = resize_cache(workspace13, (local_E * padded_M, K))
    else:
        c1 = resize_cache(workspace13, (M * topk, N * 2))
        c2 = resize_cache(workspace2, (M * topk, N))
        c3 = resize_cache(workspace13, (M * topk, K))

    if not per_act_token and (expert_map is not None or use_batched_format):
        # this is necessary to avoid imprecise scale calculation caused by
        # random data in the unused workspace. The workspace is unused when
        # this rank handles only partial tokens, or when it is batched .
        c1.fill_(0)

    cutlass_moe_mm(
        c1,
        a1q,
        w1,
        a1q_scale,
        w1_scale,
        expert_offsets,
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
        per_act_token,
        per_out_ch,
    )

    activation_callable(c2, c1, torch.cuda.current_stream().cuda_stream)

    a2q, a2q_scale = moe_kernel_quantize_input(
        c2, a2_scale, torch.float8_e4m3fn, per_act_token
    )

    if expert_map is not None:
        c3.fill_(0)

    cutlass_moe_mm(
        c3,
        a2q,
        w2,
        a2q_scale,
        w2_scale,
        expert_offsets,
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
        per_act_token,
        per_out_ch,
    )

    if use_batched_format:
        output.copy_(c3.reshape(local_E, padded_M, K), non_blocking=True)
    else:
        # We can't do this inplace because output may point to the same tensor
        # as c3.
        output.copy_(c3[c_map].view(M * topk, K), non_blocking=True)


class CutlassExpertsFp8(mm.FusedMoeExpertExecutor):
    """
    FusedMoeExpertExecutor non-batched mode mplemented based on cutlass GroupedGEMM.
    In this class, the input tokens are not padded: thus, the shape
    of the input is [total_num_tokens, hidden_size]. The input and output
    require shuffling by a_map and c_map such that the tokens assigned to
    each expert are contiguous.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
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
        assert per_out_ch_quant is False
        assert block_shape is None

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:

        topk_ids = payload.expert_topk_ids
        expert_num_tokens = (
            payload.expert_tokens_meta.expert_num_tokens if expert_map is None else None
        )
        E, N, _ = self.w1.size()
        K = self.w2.size(1)
        assert payload.expert_x.dim() == 2  # [total_num_tokens, hidden_size]
        assert topk_ids.size(0) == payload.expert_x.size(0)
        assert topk_ids.dim() == 2
        assert activation == "SiGLU"
        M = payload.expert_x.size(0)
        topk = topk_ids.size(1)

        if payload.expert_x.dtype is not torch.float8_e4m3fn:
            assert payload.expert_x.dtype == torch.bfloat16
            assert payload.expert_x_scale is None
            # per tensor quant bf16 input to fp8
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                payload.expert_x,
                None,
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=self.quant_config.is_per_act_token,
                block_shape=None,
            )
        else:
            assert payload.expert_x_scale is not None
            expert_x = payload.expert_x
            expert_x_scale = payload.expert_x_scale

        workspace13_shape = (M * topk, max(N, K))
        workspace2_shape = (M * topk, (N // 2))
        output_shape = (M * topk, K)
        workspace_dtype = payload.expert_x_origin_dtype
        workspace13 = torch.empty(
            prod(workspace13_shape),
            device=payload.expert_x.device,
            dtype=workspace_dtype,
        )
        workspace2 = torch.empty(
            prod(workspace2_shape),
            device=payload.expert_x.device,
            dtype=workspace_dtype,
        )
        output = resize_cache(workspace13, output_shape)

        activation_callable = silu_and_mul

        run_cutlass_moe_fp8(
            output,
            expert_x,
            self.w1,
            self.w2,
            topk_ids,
            activation_callable,
            self.local_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            expert_x_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_num_tokens,
            payload.expert_x_origin_dtype,
            self.quant_config.is_per_act_token,
            False,
            False,
        )
        return output


class CutlassBatchedExpertsFp8(mm.FusedMoeExpertExecutor):
    """
    FusedMoeExpertExecutor batched mode implemented based on cutlass GroupedGEMM.
    In batched mode, input tokens are padded per expert to ensure that
    the batched dispatch and combine functions work correctly: thus, the shape
    of the input is [num_experts, max_num_tokens_per_expert, hidden_size].
    The batched input and output require no shuffling by a_map and c_map since
    their tokens are already contiguous for each expert as a result of
    the dispatch function.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
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

        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self.num_local_experts = self.w1.size(0)

        assert per_out_ch_quant is False
        assert block_shape is None

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        topk_ids = payload.expert_topk_ids
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens

        E, N, _ = self.w1.size()
        K = self.w2.size(1)
        assert (
            payload.expert_x.dim() == 3
        )  # [num_experts, max_num_tokens_per_expert, hidden_size]
        assert payload.expert_x.size(0) == E
        assert topk_ids.dim() == 2
        assert activation == "SiGLU"
        M = payload.expert_x.size(1)

        if payload.expert_x.dtype is not torch.float8_e4m3fn:
            assert payload.expert_x.dtype == torch.bfloat16
            assert payload.expert_x_scale is None
            # per tensor quant bf16 input to fp8
            # Note: deepep low latency dispatch not always pad 0, this will cause an incorrect scale to be calculated here.
            if self.quant_config.is_per_tensor:
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
                    x, scale, torch.float8_e4m3fn, per_act_token, None
                )
                expert_x = q_x.view(E, -1, H)
        else:
            assert payload.expert_x_scale is not None
            expert_x = payload.expert_x
            expert_x_scale = payload.expert_x_scale

        workspace1_shape = (self.local_num_experts, M * self.num_dispatchers, max(N, K))
        workspace2_shape = (self.local_num_experts, M * self.num_dispatchers, (N // 2))
        output_shape = (self.local_num_experts, M, K)
        workspace_dtype = payload.expert_x_origin_dtype
        workspace13 = torch.empty(
            prod(workspace1_shape),
            device=payload.expert_x.device,
            dtype=workspace_dtype,
        )
        workspace2 = torch.empty(
            prod(workspace2_shape),
            device=payload.expert_x.device,
            dtype=workspace_dtype,
        )
        output = torch.empty(
            output_shape,
            device=payload.expert_x.device,
            dtype=workspace_dtype,
        )

        activation_callable = silu_and_mul

        run_cutlass_moe_fp8(
            output,
            expert_x,
            self.w1,
            self.w2,
            topk_ids,
            activation_callable,
            global_num_experts,
            None,
            self.w1_scale,
            self.w2_scale,
            expert_x_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_num_tokens,
            payload.expert_x_origin_dtype,
            self.quant_config.is_per_act_token,
            False,
            True,
        )
        return output
