from typing import Any, Dict, Optional

import aiter
import numpy as np
import torch
import torch.nn.functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import ExecutorType
from rtp_llm.utils.model_weight import W

BLOCK_SIZE_M = 32


class FusedMoeExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if FusedMoeExecutor can handle the configuration"""
        # ROCm executor doesn't have specific conditions beyond router checks
        pass

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(FusedMoEQuantConfig())

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank

        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]

        self.num_experts = config.expert_num

    def parse_sorted_ids(self, sorted_ids: torch.Tensor):
        arr_uint32 = sorted_ids.cpu().numpy().view(np.uint32)
        results = []
        for i, val in enumerate(arr_uint32):
            topk_id = (val >> 24) & 0xFF
            token_id = val & 0xFFFFFF
            results.append(
                {
                    "index": i,
                    "signed": sorted_ids[i].item(),
                    "unsigned": val,
                    "topk_id": topk_id,
                    "token_id": token_id,
                }
            )
            print(
                f"[{i:3d}] {sorted_ids[i].item():12d} -> topk_id={topk_id}, token_id={token_id}"
            )
        return results

    def get_block_size(self, num_tokens: int, topk: int, num_experts: int) -> int:
        """
        根据负载自动选择 block_size_m
        """
        avg_tokens_per_expert = (num_tokens * topk) / num_experts
        support_list = [32, 64, 128]
        for bs in support_list:
            if avg_tokens_per_expert <= bs * 4:
                return bs
        return support_list[-1]

    def moe_sorting_ck(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        model_dim: int,
        moebuf_dtype: torch.dtype,
        block_size: int = BLOCK_SIZE_M,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        调用 aiter 的 sorting kernel，对 token 按专家排序
        """
        device = topk_ids.device
        M, topk = topk_ids.shape

        # 计算 padding 后的最大长度
        max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

        # 分配输出缓冲区
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=device
        )
        sorted_weights = torch.empty(
            (max_num_tokens_padded,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

        aiter.moe_sorting_fwd(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            moe_buf=moe_buf,
            num_experts=num_experts,
            unit_size=block_size,
            local_expert_mask=expert_mask,
        )

        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:

        # === 输入验证 ===
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is None"
        assert payload.expert_topk_weights is not None, "expert_topk_weights is None"

        a1 = payload.expert_x
        topk_ids = payload.expert_topk_ids.to(torch.int32)
        topk_weights = payload.expert_topk_weights.to(torch.float32)

        M, topk = topk_ids.shape
        if M == 0:
            return torch.empty((0, a1.shape[-1]), dtype=a1.dtype, device=a1.device)

        dtype = a1.dtype
        device = topk_ids.device
        model_dim = a1.shape[-1]
        inter_dim = self.w13_weight.shape[1]

        # === 选择 block_size ===
        block_size = self.get_block_size(M, topk, self.num_experts)

        # === 构建 local_expert_mask（用于 EP）===
        expert_mask = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        num_experts_per_partition = self.num_experts // self.ep_size
        if self.ep_size > 1:
            expert_mask[0:num_experts_per_partition] = 1
        else:
            start = self.ep_rank * num_experts_per_partition
            end = start + num_experts_per_partition
            expert_mask[start:end] = 1

        # === MoE Sorting ===
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
        ) = self.moe_sorting_ck(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            model_dim=model_dim,
            moebuf_dtype=dtype,
            block_size=block_size,
            expert_mask=expert_mask,
        )

        # === Stage 1: Up/Gate Projection + Activation ===
        a2 = torch.empty((M, topk, inter_dim // 2), dtype=dtype, device=device)
        fc1_scale = None
        a1_scale = None
        # act_op = 1 if activation == "silu" else 0  # 1 = silu_and_mul

        aiter.ck_moe_stage1(
            hidden_states=a1,
            w1=self.w13_weight,
            w2=self.w2_weight,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=a2,
            topk=topk,
            kernelName="",
            w1_scale=fc1_scale,
            a1_scale=a1_scale,
            block_m=block_size,
            sorted_weights=sorted_weights if apply_router_weight_on_input else None,
        )

        # Reshape for stage2
        a2 = a2.view(M, topk, -1)

        # === Stage 2: Down Projection + Weighted Combine ===
        fc2_scale = None
        a2_scale = None

        aiter.ck_moe_stage2(
            inter_states=a2,  # [M*topk, inter_dim//2]
            w1=self.w13_weight,
            w2=self.w2_weight,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=moe_buf,  # [M, D]
            topk=topk,
            kernelName="",
            w2_scale=fc2_scale,
            a2_scale=a2_scale,
            block_m=block_size,
            sorted_weights=sorted_weights if not apply_router_weight_on_input else None,
        )

        return moe_buf


def torch_moe_ref(
    payload: ExpertForwardPayload,
    activation: str,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
    extra_expert_args: Optional[Dict[str, Any]],
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of MoE forward for testing.
    默认不启用量化，所有量化参数可为 None。

    Args:
        payload: 输入数据包
        activation: "silu" 或 "gelu"
        global_num_experts: 总专家数
        expert_map: （保留，暂未使用）
        a2_scale: （保留，ref 中暂不使用）
        apply_router_weight_on_input: 是否在输入时加权（本实现支持两种模式）
        extra_expert_args: 扩展参数（保留）

    Returns:
        torch.Tensor: [M, D] MoE 输出
    """
    # === 提取输入 ===
    hidden_states = payload.expert_x  # [M, D]
    topk_ids = payload.expert_topk_ids  # [M, top_k]
    topk_weights = payload.expert_topk_weights  # [M, top_k]

    M, D = hidden_states.shape
    top_k = topk_ids.shape[1]

    if M == 0:
        return torch.empty(
            (0, D), dtype=hidden_states.dtype, device=hidden_states.device
        )

    # 使用输入精度（支持 bf16/fp16）
    compute_dtype = hidden_states.dtype
    w1 = w1.to(compute_dtype)
    w2 = w2.to(compute_dtype)
    hidden_states = hidden_states.to(compute_dtype)

    # === 扩展输入：[M, D] -> [M, top_k, D] ===
    hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1)  # [M, top_k, D]

    # 存放每个 (token, topk) 的 FFN 输出
    ffn_out = torch.zeros(
        (M, top_k, D),
        dtype=compute_dtype,
        device=hidden_states.device,
    )

    # === 遍历每个专家 ===
    for expert_id in range(global_num_experts):
        mask = topk_ids == expert_id  # [M, top_k]
        if not mask.any():
            continue
        # print("expert_id: ", expert_id)
        # print("mask: ", mask)

        # 获取该专家处理的 token 输入
        tokens = hidden_expanded[mask]  # [S, D]
        # print("tokens shape: ", tokens.shape)
        # print("w1[expert_id].t(): ", w1[expert_id].t())

        # Step 1: Up/Gate Projection
        upgate = torch.matmul(tokens, w1[expert_id].t())  # [S, N]
        # print("upgate: ", upgate)

        # Step 2: 激活函数
        gate, up = upgate.chunk(2, dim=-1)
        # print("gate: ", gate)
        # print("up: ", up)
        if activation == "silu":
            activated = F.silu(gate) * up
        elif activation == "gelu":
            activated = F.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Step 3: Down Projection
        # print("activated: ", activated)
        # print("w2[expert_id].t(): ", w2[expert_id].t())
        out = torch.matmul(activated, w2[expert_id].t())  # [S, D]
        # print("out: ", out)

        # 写回
        ffn_out[mask] = out

    # === 加权求和 ===
    if apply_router_weight_on_input == False:
        weighted = ffn_out * topk_weights.unsqueeze(-1)

    final_out = weighted.sum(dim=1)  # [M, D]
    # print("final_out: ", final_out)

    return final_out.to(hidden_states.dtype)  # 保持输出精度一致
