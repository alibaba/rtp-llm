"""MiniMax-M3 for the new loader.

Hybrid Dense/MoE model:
  - 60 Transformer layers
  - Layers 0-2: Dense FFN (intermediate=12288)
  - Layers 3-59: MoE (128 experts, top-4, + 1 shared expert, intermediate=3072)
  - Attention: GQA (64 Q heads / 4 KV heads, head_dim=128) + QK Norm
  - Sparse attention index projections (layers 3-59)
  - Quantization: MXFP8 (weight_scale_inv, block size [1,32])

Supports BF16 / MXFP8 ckpts with TP and EP parallelism.

Submodule names match checkpoint paths exactly so RtpModule.load_weights
dispatches weights via recursive name matching without any fusion-time
mapping.  The ``WEIGHTS_MAPPER`` strips the ``model.`` prefix that the
HuggingFace ckpt carries.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.activation import swigluoai_and_mul
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import GroupTopK, MultimodalEmbeddingInjector, SelectTopk
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs

logger = logging.getLogger(__name__)


class MiniMaxM3GemmaRMSNorm(RMSNorm):
    """MiniMax-M3 uses Gemma RMSNorm: checkpoint weights are stored pre +1."""

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                self.weight.data.copy_(tensor + 1)


# ------------------------------------------------------------------ #
#  Routed experts
# ------------------------------------------------------------------ #


class MiniMaxM3Experts(BaseMoEExperts):
    """MiniMax-M3 routed-expert module.

    Checkpoint names expert projections w1 / w3 / w2 (gate / up / down).
    ``BaseMoEExperts`` stores the first w13 half through the ``up_proj`` path
    and the second half through ``gate_proj``; map M3's names to those semantic
    slots and select the SwiGLU-OAI activation in fused MoE.

    MXFP8 quantization is mapped to the ``fp8_per_block`` family so the
    built-in block-scale dispatch (``weight_scale_inv``) is inherited.
    """

    PROJ_NAMES = ("w1", "w3", "w2")

    _EXTRA_QUANT_MAP = {
        "mxfp8": "fp8_per_block",
    }

    def _dispatch_weight(self, local_id: int, proj: str, tensor: torch.Tensor):
        if proj == "w1":
            proj = "gate_proj"
        elif proj == "w3":
            proj = "up_proj"
        elif proj == "w2":
            proj = "down_proj"
        super()._dispatch_weight(local_id, proj, tensor)

    def _dispatch_scale(
        self, local_id: int, proj: str, param_name: str, tensor: torch.Tensor
    ):
        if proj == "w1":
            proj = "gate_proj"
        elif proj == "w3":
            proj = "up_proj"
        elif proj == "w2":
            proj = "down_proj"
        super()._dispatch_scale(local_id, proj, param_name, tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.fused_moe is None:
            self._maybe_build_fused_moe()
        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="swigluoai",
        )


# ------------------------------------------------------------------ #
#  Shared expert
# ------------------------------------------------------------------ #


class MiniMaxM3SharedExpert(RtpModule):
    """Shared expert FFN (always activated, no gating).

    HF ckpt keys:
      block_sparse_moe.shared_experts.gate_proj.weight
      block_sparse_moe.shared_experts.up_proj.weight
      block_sparse_moe.shared_experts.down_proj.weight
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
        x = swigluoai_and_mul(gate_up)
        x = self.down_proj(x)
        return x


# ------------------------------------------------------------------ #
#  MoE block
# ------------------------------------------------------------------ #


class MiniMaxM3MoeBlock(RtpModule):
    """Routed-expert MoE block with shared expert.

    HF ckpt keys:
      block_sparse_moe.gate.weight              (F32, no bias)
      block_sparse_moe.e_score_correction_bias   (F32)
      block_sparse_moe.experts.{j}.w1/w3/w2.weight + .weight_scale_inv
      block_sparse_moe.shared_experts.{gate_proj|up_proj|down_proj}.weight
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
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
        routed_scaling_factor: float = 2.0,
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        # Router: hidden -> num_experts, NOT TP-sharded, always F32.
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_experts,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="gate",
            bias=False,
            params_dtype=torch.float32,
        )
        # e_score_correction_bias: [num_experts] F32, loaded via streaming dispatch.
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.select_topk = SelectTopk(config=model_config)

        # Routed experts
        self.experts = MiniMaxM3Experts(
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

        # Shared expert (named "shared_experts" to match ckpt)
        self.shared_experts = MiniMaxM3SharedExpert(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        router_logits = self.gate(hidden_states.float())
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

        # Use GroupTopK with e_score_correction_bias (MiniMax-M3 always has it).
        group_topk = GroupTopK()
        group_topk(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            scores=router_logits_fp32,
            correction_bias=self.e_score_correction_bias,
            n_group=self.n_group,
            topk_group=self.topk_group,
            topk=self.top_k,
            renormalize=True,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        routed = self.experts(hidden_states, topk_weights, topk_ids)
        shared = self.shared_experts(hidden_states)

        if self.ep_size > 1:
            # EP: routed output is complete; shared is TP-partial.
            shared = all_reduce(shared, group=Group.TP)

        return routed + shared


# ------------------------------------------------------------------ #
#  Attention
# ------------------------------------------------------------------ #


class MiniMaxM3Attention(RtpModule):
    """MiniMax-M3 GQA attention with QK Norm and optional index projections.

    Uses separate q_proj / k_proj / v_proj (not QKVParallelLinear) because
    the TP split dimensions differ (Q: num_heads, KV: num_kv_heads).

    Sparse-attention index projections (index_q_proj / index_k_proj) are
    created for layers where ``has_index_projs=True`` (layers 3-59) so the
    checkpoint weights can be loaded.  They are not used in ``forward`` yet.

    HF ckpt keys:
      self_attn.q_proj.weight / .weight_scale_inv
      self_attn.k_proj.weight / .weight_scale_inv
      self_attn.v_proj.weight / .weight_scale_inv
      self_attn.o_proj.weight / .weight_scale_inv
      self_attn.q_norm.weight
      self_attn.k_norm.weight
      self_attn.index_q_proj.weight   (layers 3-59 only)
      self_attn.index_k_proj.weight   (layers 3-59 only)
      self_attn.index_q_norm.weight   (layers 3-59 only)
      self_attn.index_k_norm.weight   (layers 3-59 only)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        rms_norm_eps: float = 1e-6,
        has_index_projs: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        self.q_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="q_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.k_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="k_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.v_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="v_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.q_norm = MiniMaxM3GemmaRMSNorm(
            head_dim, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.k_norm = MiniMaxM3GemmaRMSNorm(
            head_dim, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="o_proj",
            bias=False,
            params_dtype=params_dtype,
        )

        # Sparse attention index projections (layers 3-59).
        # Created to match ckpt names; not used in forward yet.
        self.has_index_projs = has_index_projs
        if has_index_projs:
            # index_q: 4 heads * head_dim = 512
            self.index_q_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=num_kv_heads * head_dim,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                prefix="index_q_proj",
                bias=False,
                params_dtype=params_dtype,
            )
            # index_k: 1 * head_dim = 128
            self.index_k_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=head_dim,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                prefix="index_k_proj",
                bias=False,
                params_dtype=params_dtype,
            )
            self.index_q_norm = MiniMaxM3GemmaRMSNorm(
                head_dim, eps=rms_norm_eps, params_dtype=params_dtype
            )
            self.index_k_norm = MiniMaxM3GemmaRMSNorm(
                head_dim, eps=rms_norm_eps, params_dtype=params_dtype
            )

    def _apply_qk_norm(self, qkv: torch.Tensor) -> torch.Tensor:
        prefix_shape = qkv.shape[:-1]
        q = qkv[..., : self.q_size].reshape(
            *prefix_shape, self.num_heads_per_partition, self.head_dim
        )
        k = qkv[..., self.q_size : self.q_size + self.kv_size].reshape(
            *prefix_shape, self.num_kv_heads_per_partition, self.head_dim
        )
        v = qkv[..., self.q_size + self.kv_size :]
        q = self.q_norm(q).reshape(*prefix_shape, self.q_size)
        k = self.k_norm(k).reshape(*prefix_shape, self.kv_size)
        return torch.cat([q, k, v], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # Concatenate into a single [q, k, v] tensor matching QKVParallelLinear layout.
        qkv = torch.cat([q, k, v], dim=-1)
        qkv = self._apply_qk_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


# ------------------------------------------------------------------ #
#  Dense FFN (layers 0-2)
# ------------------------------------------------------------------ #


class MiniMaxM3DenseFFN(RtpModule):
    """Dense FFN for the first 3 layers.

    HF ckpt keys:
      mlp.gate_proj.weight / .weight_scale_inv
      mlp.up_proj.weight   / .weight_scale_inv
      mlp.down_proj.weight / .weight_scale_inv
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
        act = swigluoai_and_mul(gate_up)
        x = self.down_proj(act)
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


# ------------------------------------------------------------------ #
#  Decoder layer
# ------------------------------------------------------------------ #


class MiniMaxM3DecoderLayer(RtpModule):
    """MiniMax-M3 decoder layer: attention + Dense/MoE FFN.

    Module naming:
      - MoE layers: ``self.block_sparse_moe`` (matches ckpt)
      - Dense layers: ``self.mlp`` (matches ckpt)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dense_intermediate_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
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
        rms_norm_eps: float,
        is_moe: bool,
        has_index_projs: bool = False,
        routed_scaling_factor: float = 2.0,
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.is_moe = is_moe
        self.input_layernorm = MiniMaxM3GemmaRMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = MiniMaxM3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            rms_norm_eps=rms_norm_eps,
            has_index_projs=has_index_projs,
        )
        self.post_attention_layernorm = MiniMaxM3GemmaRMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        if is_moe:
            self.block_sparse_moe = MiniMaxM3MoeBlock(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                shared_expert_intermediate_size=shared_expert_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                layer_idx=layer_idx,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=quant_config,
                params_dtype=params_dtype,
                routed_scaling_factor=routed_scaling_factor,
                n_group=n_group,
                topk_group=topk_group,
            )
        else:
            self.mlp = MiniMaxM3DenseFFN(
                hidden_size=hidden_size,
                intermediate_size=dense_intermediate_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states)
            if self.block_sparse_moe.ep_size <= 1 and self.block_sparse_moe.tp_size > 1:
                # TP-only: MoE FFN inner-dim is TP-sharded; reduce across TP ranks.
                # EP: FusedMoe handles EP combine internally.
                hidden_states = all_reduce(hidden_states, group=Group.TP)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


# ------------------------------------------------------------------ #
#  Config extraction
# ------------------------------------------------------------------ #


def _extract_config_values(model_config: Any, load_config: Any) -> Dict[str, Any]:
    """Pull all fields needed to build MiniMaxM3 layers.

    Handles both:
      - C++ pybind ModelConfig (flat, legacy field names)
      - Raw HF config dict (possibly with nested ``text_config``)
    """

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # Resolve nested text_config for HF dict configs.
    text_config = _get(model_config, "text_config", None)
    if text_config is None:
        qc = _get(model_config, "quant_config", None)
        if qc is not None:
            text_config = getattr(qc, "text_config", None)

    def _get_text(name, default=None):
        """Look up in text_config first, then model_config."""
        if text_config is not None:
            val = _get(text_config, name, None)
            if val is not None:
                return val
        return _get(model_config, name, default)

    hidden_size = _get_text("hidden_size", 6144)
    num_layers = _get_text("num_layers", _get_text("num_hidden_layers", 60))
    vocab_size = _get_text("vocab_size", 200064)

    # Attention
    attn_config = _get_text("attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 64)
        num_kv_heads = _get(attn_config, "kv_head_num", num_heads)
        head_dim = _get(attn_config, "size_per_head", hidden_size // num_heads)
    else:
        num_heads = _get_text("num_attention_heads", 64)
        num_kv_heads = _get_text("num_key_value_heads", num_heads)
        head_dim = _get_text("head_dim", 128)

    rms_norm_eps = _get_text("layernorm_eps", _get_text("rms_norm_eps", 1e-6))

    # Dense FFN intermediate (layers 0-2)
    dense_intermediate_size = _get_text("dense_intermediate_size", 12288)

    # MoE intermediate (routed experts)
    moe_intermediate_size = _get_text(
        "moe_inter_size",
        _get_text("moe_intermediate_size", _get_text("intermediate_size", 3072)),
    )

    # Shared expert intermediate
    shared_expert_intermediate_size = _get_text(
        "shared_expert_intermediate_size",
        _get_text("shared_intermediate_size", moe_intermediate_size),
    )

    # Experts
    num_experts = _get_text(
        "expert_num", _get_text("num_local_experts", _get_text("num_experts", 128))
    )
    top_k = _get_text("moe_k", _get_text("num_experts_per_tok", 4))

    # MoE layer frequency: list of 0/1 per layer (1 = MoE, 0 = Dense).
    moe_layer_freq = _get_text("moe_layer_freq", None)
    if moe_layer_freq is not None and not isinstance(moe_layer_freq, list):
        first_k_dense = int(moe_layer_freq)
        moe_layer_freq = [0 if i < first_k_dense else 1 for i in range(num_layers)]
    if moe_layer_freq is None:
        # Default: first 3 layers dense, rest MoE
        moe_layer_freq = [0, 0, 0] + [1] * (num_layers - 3)

    # Sparse attention frequency: list of 0/1 per layer.
    sparse_attn_config = _get_text("sparse_attention_config", None)
    sparse_attention_freq = _get_text("sparse_attention_freq", None)
    if sparse_attention_freq is None:
        if sparse_attn_config is not None:
            sparse_attention_freq = _get(
                sparse_attn_config, "sparse_attention_freq", None
            )
    if sparse_attention_freq is not None and not isinstance(
        sparse_attention_freq, list
    ):
        sparse_attention_freq = None
    if sparse_attention_freq is None:
        # Default: same pattern as MoE (layers 3-59)
        sparse_attention_freq = [0, 0, 0] + [1] * (num_layers - 3)

    # Index projection dims (for sparse attention, layers 3-59)
    index_q_dim = num_kv_heads * head_dim  # 4 * 128 = 512
    index_k_dim = head_dim  # 128
    if sparse_attn_config is not None:
        index_q_dim = _get(sparse_attn_config, "index_q_dim", index_q_dim)
        index_k_dim = _get(sparse_attn_config, "index_k_dim", index_k_dim)

    # Routing
    routed_scaling_factor = _get_text("routed_scaling_factor", 2.0)
    if sparse_attn_config is not None:
        routed_scaling_factor = _get(
            sparse_attn_config, "routed_scaling_factor", routed_scaling_factor
        )

    # Group routing (GroupTopK)
    n_group = _get_text("n_group", 1)
    topk_group = _get_text("topk_group", 1)

    # Parallelism
    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    ep_size = getattr(load_config, "ep_size", 1)
    ep_rank = getattr(load_config, "ep_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.bfloat16)
    parallelism_config = getattr(load_config, "parallelism_config", None)
    moe_config = getattr(load_config, "moe_config", None)

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        dense_intermediate_size=dense_intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        moe_layer_freq=moe_layer_freq,
        sparse_attention_freq=sparse_attention_freq,
        index_q_dim=index_q_dim,
        index_k_dim=index_k_dim,
        routed_scaling_factor=routed_scaling_factor,
        n_group=n_group,
        topk_group=topk_group,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class MiniMaxM3ForCausalLM(GptModelBase):
    """MiniMax-M3 hybrid Dense/MoE model for the new loader.

    ``WEIGHTS_MAPPER`` strips the ``model.`` prefix that the HF ckpt carries.
    All submodule names match ckpt keys directly, so ``RtpModule.load_weights``
    dispatches weights without any fusion-time mapping.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        if isinstance(weights, dict):
            if len(weights) == 1:
                mapped_iter = self.WEIGHTS_MAPPER.apply(iter(weights.items()))
                super().load_weights(mapped_iter)
                return
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head:
            logger.info(
                "[MiniMaxM3ForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

        cfg = _extract_config_values(model_config, load_config)
        moe_layer_set = {i for i, v in enumerate(cfg["moe_layer_freq"]) if v}
        sparse_attn_set = {i for i, v in enumerate(cfg["sparse_attention_freq"]) if v}

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    head_dim=cfg["head_dim"],
                    dense_intermediate_size=cfg["dense_intermediate_size"],
                    moe_intermediate_size=cfg["moe_intermediate_size"],
                    shared_expert_intermediate_size=cfg[
                        "shared_expert_intermediate_size"
                    ],
                    num_experts=cfg["num_experts"],
                    top_k=cfg["top_k"],
                    layer_idx=i,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    ep_size=cfg["ep_size"],
                    ep_rank=cfg["ep_rank"],
                    model_config=cfg["model_config"],
                    parallelism_config=cfg["parallelism_config"],
                    moe_config=cfg["moe_config"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                    is_moe=(i in moe_layer_set),
                    has_index_projs=(i in sparse_attn_set),
                    routed_scaling_factor=cfg["routed_scaling_factor"],
                    n_group=cfg["n_group"],
                    topk_group=cfg["topk_group"],
                )
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = MiniMaxM3GemmaRMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        mm = inputs.multimodal_inputs
        if mm is not None and mm.multimodal_features:
            hidden_states = self.multimodal_embedding_injector(
                inputs_embeds, mm.multimodal_features, mm.mm_features_locs
            )
        else:
            hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
