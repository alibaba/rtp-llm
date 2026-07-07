"""
MoE layers for DeepSeek V3.2, new-loader style.

Key design:
  - DeepSeekV32Experts extends BaseMoEExperts for the routed experts.
    HF ckpt provides per-expert weights (gate_proj, up_proj, down_proj) which
    BaseMoEExperts handles with its own EP/TP streaming load_weights override
    (taking precedence over RtpModule's default via normal Python MRO).
  - DeepSeekV32MoEBlock wraps gate + SelectTopk(/GroupTopK) + experts + shared_expert.
    Mirrors GenericMoeLayer.forward but with new-loader submodules.
  - Shared expert is a plain DenseMLP (gate_up → SiLU*gate → down) using
    MergedColumnParallelLinear + RowParallelLinear.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import GroupTopK, SelectTopk
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class DeepSeekV32Experts(BaseMoEExperts):
    """Routed experts for DeepSeek V3.2.

    Inherits everything from BaseMoEExperts:
      - EP/TP expert loading via _dispatch_weight / _dispatch_scale
      - FP8/FP4 scale fusion in process_weights_after_loading
      - _build_weights_dict → FusedMoeFactory

    Quantization is driven by the LoadConfig's quant_type. An already-quantized
    FP8-per-block DeepSeek-V3.2 ckpt resolves to "fp8_block"; the model then
    redirects its NON-expert linears to "fp8_block_dequant" (bf16) — see
    language._extract_config_values — and that same quant_config reaches the
    experts here. Experts must stay FP8 (DeepGEMM), so map "fp8_block_dequant"
    back to the "fp8_per_block" family. "fp8_block" (had it not been redirected)
    is already handled by BaseMoEExperts._BASE_QUANT_MAP; a plain BF16 ckpt
    resolves to "none" and loads unquantized.
    """

    _EXTRA_QUANT_MAP = {
        "fp8_block_dequant": "fp8_per_block",
    }


class DeepSeekV32MoeGate(RtpModule):
    """Router gate that owns both `weight` and `e_score_correction_bias`.

    Matches HF ckpt keys:
      model.layers.{i}.mlp.gate.weight                  -> weight
      model.layers.{i}.mlp.gate.e_score_correction_bias -> e_score_correction_bias

    Not TP-sharded (num_experts is small relative to hidden dim).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        has_correction_bias: bool,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        if has_correction_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, None)


class DeepSeekV32SharedExpertMLP(RtpModule):
    """Shared expert MLP (dense), new-loader style.

    HF ckpt keys:
      model.layers.{i}.mlp.shared_experts.gate_proj.weight
      model.layers.{i}.mlp.shared_experts.up_proj.weight
      model.layers.{i}.mlp.shared_experts.down_proj.weight
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.hidden_size = hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate_up_proj",
            bias=False,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x


class DeepSeekV32MoEBlock(RtpModule):
    """Full MoE block: gate + router + routed experts + shared expert.

    Mirrors GenericMoeLayer.forward: routing → FusedMoe → (optional) shared expert.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        # Config fields for GroupTopK (DeepSeek V3 style)
        has_shared_expert: bool = True,
        shared_expert_intermediate_size: int = 0,
        scoring_func: int = 1,  # 0=softmax, 1=sigmoid
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        has_moe_norm: bool = False,
        correction_bias: bool = False,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        self.has_shared_expert = has_shared_expert
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.has_moe_norm = has_moe_norm
        self.correction_bias = correction_bias
        self.num_experts = num_experts

        # Router gate: hidden → num_experts (not TP-sharded, small).
        # Custom wrapper owns `weight` AND `e_score_correction_bias` so the
        # HF key model.layers.{i}.mlp.gate.e_score_correction_bias loads
        # cleanly via streaming dispatch.
        self.gate = DeepSeekV32MoeGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            has_correction_bias=correction_bias,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)

        # Routed experts
        self.experts = DeepSeekV32Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=params_dtype,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
        )

        # Shared expert. HF key is `mlp.shared_experts.{gate,up,down}_proj`
        # (PLURAL). DeepSeek-V3.2 has no shared-expert gating — the routed
        # MoE output is summed directly with shared_experts(hidden_states).
        if has_shared_expert and shared_expert_intermediate_size > 0:
            self.shared_experts = DeepSeekV32SharedExpertMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids_dtype = (
            self.experts.fused_moe.topk_ids_dtype
            if self.experts.fused_moe is not None
            else torch.int32
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias:
            group_topk = GroupTopK()
            group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.gate.e_score_correction_bias,
                n_group=self.n_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.has_moe_norm,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        experts_output = self.experts(hidden_states, topk_weights, topk_ids)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            if self.tp_size > 1 and self.ep_size > 1:
                shared_output = all_reduce(shared_output, group=Group.TP)
            experts_output = experts_output + shared_output

        return experts_output
