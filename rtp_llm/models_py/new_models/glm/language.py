"""ChatGLM2/3/4 dense model for the new (vLLM-style) weight loader.

Faithfully ports the legacy `rtp_llm/models/glm_v2_weight.py` (GlmV2WeightInfo)
checkpoint layout (THUDM ChatGLM2/3/4):

  transformer.embedding.word_embeddings.weight              -> embed_tokens
  transformer.encoder.layers.{i}.input_layernorm.weight     -> input_layernorm (RMSNorm)
  transformer.encoder.layers.{i}.self_attention.query_key_value.{weight,bias}
                                                            -> FUSED qkv (+bias), split q/k/v
  transformer.encoder.layers.{i}.self_attention.dense.weight-> o_proj (no bias)
  transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight   -> FUSED [gate; up], split
  transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight   -> down_proj (no bias)
  transformer.encoder.layers.{i}.post_attention_layernorm.weight -> post_attention_layernorm
  transformer.encoder.final_layernorm.weight                -> norm
  transformer.output_layer.weight                           -> lm_head (NOT tied)

GLM specifics vs llama/qwen:
  * qkv has a fused bias (split into q/k/v).
  * qkv / gate_up arrive as SINGLE fused tensors -> custom load_weights splits
    them and re-dispatches as q_proj/k_proj/v_proj and gate_proj/up_proj so the
    standard QKV/MergedColumn linears can do their TP slicing.
  * GQA via multi_query_group_num (e.g. chatglm2-6b: 32 q heads, 2 kv heads).
  * RoPE is GLM2 style (RopeStyle.Glm2 == 2, partial+interleaved). The model
    config already carries style=2; the rtp fused-rope kernel (used by trt/xqa/
    headwise fmha impls) applies it. The flashinfer-prefill rope path does NOT
    support glm2 — if forward looks wrong, force a fused-rope fmha impl.
  * Residual structure is standard pre-LN (same as llama): input_layernorm ->
    attn -> +residual -> post_attention_layernorm -> mlp -> +residual.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import GroupTopK, SelectTopk
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs


class GlmMLP(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.tp_size = tp_size
        # dense_h_to_4h packs [gate; up]; we split it in the model's load_weights
        # and feed the two halves here as gate_proj / up_proj shards.
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
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


class GlmAttention(RtpModule):

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
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size

        # GLM ships a fused query_key_value with bias; o_proj (dense) has no bias.
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="qkv_proj",
            bias=True,
            params_dtype=params_dtype,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


class GlmDecoderLayer(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        layer_idx: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = GlmAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.mlp = GlmMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _extract_config_values(model_config: Any, load_config: Any):
    if isinstance(model_config, dict):
        hidden_size = model_config.get("hidden_size", 4096)
        num_heads = model_config.get("num_attention_heads", 32)
        num_kv_heads = model_config.get("multi_query_group_num", num_heads)
        intermediate_size = model_config.get("ffn_hidden_size", 13696)
        num_layers = model_config.get("num_layers", 28)
        vocab_size = model_config.get("padded_vocab_size", 65024)
        head_dim = hidden_size // num_heads
        rms_norm_eps = model_config.get("layernorm_epsilon", 1e-5)
    else:
        hidden_size = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(
            model_config, "num_layers", getattr(model_config, "num_hidden_layers", 28)
        )
        vocab_size = getattr(model_config, "vocab_size", 65024)
        intermediate_size = getattr(
            model_config,
            "inter_size",
            getattr(model_config, "intermediate_size", 13696),
        )
        attn_config = getattr(model_config, "attn_config", None)
        if attn_config is not None:
            num_heads = getattr(attn_config, "head_num", 32)
            num_kv_heads = getattr(attn_config, "kv_head_num", num_heads)
            head_dim = getattr(attn_config, "size_per_head", hidden_size // num_heads)
        else:
            num_heads = getattr(model_config, "num_attention_heads", 32)
            num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
            head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)
        rms_norm_eps = getattr(
            model_config,
            "layernorm_eps",
            getattr(model_config, "rms_norm_eps", 1e-5),
        )

    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.float16)

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        tp_size=tp_size,
        tp_rank=tp_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
    )


class ChatGLMForCausalLM(GptModelBase):

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

        cfg = _extract_config_values(model_config, load_config)
        self._num_heads = cfg["num_heads"]
        self._num_kv_heads = cfg["num_kv_heads"]
        self._head_dim = cfg["head_dim"]
        self._inter_size = cfg["intermediate_size"]

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                GlmDecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    intermediate_size=cfg["intermediate_size"],
                    head_dim=cfg["head_dim"],
                    layer_idx=i,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                )
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSNorm(
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

    # ------------------------------------------------------------------ #
    #  Weight loading: rename GLM ckpt names + split fused qkv / gate_up
    # ------------------------------------------------------------------ #
    def _rewrite(
        self, name: str, tensor: torch.Tensor
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        # Global tensors.
        if name == "transformer.embedding.word_embeddings.weight":
            yield "embed_tokens.weight", tensor
            return
        if name == "transformer.encoder.final_layernorm.weight":
            yield "norm.weight", tensor
            return
        if name == "transformer.output_layer.weight":
            yield "lm_head.weight", tensor
            return

        prefix = "transformer.encoder.layers."
        if not name.startswith(prefix):
            # Unknown global (e.g. prefix_encoder / rotary buffers) — drop.
            return
        rest = name[len(prefix) :]
        layer_idx, sub = rest.split(".", 1)
        base = f"layers.{layer_idx}"

        if sub == "input_layernorm.weight":
            yield f"{base}.input_layernorm.weight", tensor
        elif sub == "post_attention_layernorm.weight":
            yield f"{base}.post_attention_layernorm.weight", tensor
        elif sub == "self_attention.dense.weight":
            yield f"{base}.self_attn.o_proj.weight", tensor
        elif sub == "mlp.dense_4h_to_h.weight":
            yield f"{base}.mlp.down_proj.weight", tensor
        elif sub in (
            "self_attention.query_key_value.weight",
            "self_attention.query_key_value.bias",
        ):
            # Fused qkv: rows [q | k | v] -> split into q/k/v shard names so the
            # attention module's redirect maps them onto qkv_proj (same contract
            # as qwen3's separate q_proj/k_proj/v_proj ckpt tensors).
            q = self._num_heads * self._head_dim
            kv = self._num_kv_heads * self._head_dim
            param = "weight" if sub.endswith(".weight") else "bias"
            yield f"{base}.self_attn.q_proj.{param}", tensor[0:q]
            yield f"{base}.self_attn.k_proj.{param}", tensor[q : q + kv]
            yield f"{base}.self_attn.v_proj.{param}", tensor[q + kv : q + 2 * kv]
        elif sub == "mlp.dense_h_to_4h.weight":
            # Fused [gate; up] -> split at inter_size into gate_proj/up_proj shard
            # names; the mlp module's redirect maps them onto gate_up_proj.
            inter = self._inter_size
            yield f"{base}.mlp.gate_proj.weight", tensor[0:inter]
            yield f"{base}.mlp.up_proj.weight", tensor[inter : 2 * inter]
        # else: drop (e.g. rotary_emb.inv_freq has no encoder.layers prefix anyway)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights = iter(weights.items())

        def _stream():
            for name, tensor in weights:
                yield from self._rewrite(name, tensor)

        super().load_weights(_stream())

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
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


# ------------------------------------------------------------------ #
#  GLM4-MoE
# ------------------------------------------------------------------ #

"""GLM4-MoE for the new weight loader.

HF checkpoint layout (model. prefix stripped by WEIGHTS_MAPPER):
  model.embed_tokens.weight                                 -> embed_tokens
  model.norm.weight                                         -> norm
  lm_head.weight                                            -> lm_head (NOT tied by default)
  model.layers.{i}.input_layernorm.weight                   -> input_layernorm
  model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}    -> FUSED qkv (+bias)
  model.layers.{i}.self_attn.o_proj.weight                  -> o_proj (no bias)
  model.layers.{i}.self_attn.{q,k}_norm.weight              -> qk_norm (optional)
  model.layers.{i}.post_attention_layernorm.weight         -> post_attention_layernorm

Dense layers (i < first_k_dense_replace):
  model.layers.{i}.mlp.{gate,up,down}_proj.weight

MoE layers (i >= first_k_dense_replace):
  model.layers.{i}.mlp.gate.weight                          -> router
  model.layers.{i}.mlp.gate.e_score_correction_bias         -> correction bias (optional)
  model.layers.{i}.mlp.experts.{id}.{gate,up,down}_proj.weight -> routed experts
  model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight -> shared expert

GLM4-MoE specifics:
  * Separate q/k/v projections (with bias) — NOT the fused query_key_value
    of GLM dense. After WEIGHTS_MAPPER strips model., the shard names
    q_proj/k_proj/v_proj redirect into QKVParallelLinear automatically.
  * Optional per-head qk_norm (use_qk_norm in config.json).
  * Hybrid MoE: first first_k_dense_replace layers are dense, rest are MoE.
  * Shared expert + routed experts (moe_style=2).
  * Router scoring is sigmoid (scoring_func=1), handled in forward.
"""


def _read_config_json(ckpt_path: str) -> Dict[str, Any]:
    """Read config.json from ckpt path, return empty dict if not found."""
    if not ckpt_path:
        return {}
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.loads(f.read())


class Glm4MoeAttention(RtpModule):
    """GLM4-MoE attention: fused QKV with bias + optional per-head qk_norm.

    HF ckpt keys (after model. prefix strip):
      layers.{i}.self_attn.q_proj.{weight,bias}
      layers.{i}.self_attn.k_proj.{weight,bias}
      layers.{i}.self_attn.v_proj.{weight,bias}
      layers.{i}.self_attn.o_proj.weight
      layers.{i}.self_attn.q_norm.weight  (if use_qk_norm)
      layers.{i}.self_attn.k_norm.weight  (if use_qk_norm)
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
        params_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.use_qk_norm = use_qk_norm

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="qkv_proj",
            bias=True,
            params_dtype=params_dtype,
        )
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
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
        qkv = self.qkv_proj(hidden_states)
        if self.use_qk_norm:
            qkv = self._apply_qk_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


class Glm4MoeSharedExpertMLP(RtpModule):
    """Shared expert MLP (dense) for GLM4-MoE.

    HF ckpt keys:
      layers.{i}.mlp.shared_experts.gate_proj.weight
      layers.{i}.mlp.shared_experts.up_proj.weight
      layers.{i}.mlp.shared_experts.down_proj.weight

    Does NOT do all_reduce in forward — the MoE block / decoder layer handles
    TP reduction at the appropriate level (shared output is summed with the
    routed-experts output before the layer-level all_reduce).
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
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        return x


class Glm4MoeExperts(BaseMoEExperts):
    """Routed experts for GLM4-MoE.

    Inherits EP/TP expert loading, FP8/FP4 quantization, and FusedMoeFactory
    construction from BaseMoEExperts. No additional overrides needed.
    """

    pass


class Glm4MoeGate(RtpModule):
    """Router gate that owns both ``weight`` and ``e_score_correction_bias``.

    Matches HF ckpt keys:
      layers.{i}.mlp.gate.weight
      layers.{i}.mlp.gate.e_score_correction_bias  (optional)

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


class Glm4MoeBlock(RtpModule):
    """Full MoE block: gate + router + routed experts + shared expert.

    Mirrors DeepSeekV32MoEBlock but for GLM4-MoE (standard attention, not MLA).
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
        shared_expert_intermediate_size: int = 0,
        has_shared_expert: bool = True,
        correction_bias: bool = False,
        scoring_func: int = 1,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        has_moe_norm: bool = False,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        self.correction_bias = correction_bias
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.has_moe_norm = has_moe_norm
        self.num_experts = num_experts

        self.gate = Glm4MoeGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            has_correction_bias=correction_bias,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)

        self.experts = Glm4MoeExperts(
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

        if has_shared_expert and shared_expert_intermediate_size > 0:
            self.shared_experts = Glm4MoeSharedExpertMLP(
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


class Glm4MoeDecoderLayer(RtpModule):
    """Decoder layer that can be either dense (GlmMLP) or MoE (Glm4MoeBlock)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        rms_norm_eps: float,
        use_qk_norm: bool,
        is_moe_layer: bool,
        dense_intermediate_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        scoring_func: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        has_moe_norm: bool,
        correction_bias: bool,
    ):
        super().__init__()
        self.is_moe_layer = is_moe_layer
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = Glm4MoeAttention(
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
            use_qk_norm=use_qk_norm,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        if is_moe_layer:
            self.mlp = Glm4MoeBlock(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
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
                shared_expert_intermediate_size=shared_expert_intermediate_size,
                has_shared_expert=True,
                correction_bias=correction_bias,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                n_group=n_group,
                topk_group=topk_group,
                has_moe_norm=has_moe_norm,
            )
        else:
            self.mlp = GlmMLP(
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
        hidden_states = self.mlp(hidden_states)
        if self.is_moe_layer:
            # TP-only mode: MoE FFN inner-dim is TP-sharded; reduce across TP.
            # EP mode: FusedMoe executor handles EP combine internally.
            if self.mlp.ep_size <= 1 and self.mlp.tp_size > 1:
                hidden_states = all_reduce(hidden_states, group=Group.TP)
        hidden_states = residual + hidden_states
        return hidden_states


def _extract_glm4_moe_config_values(
    model_config: Any, load_config: Any, config_json: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Pull all fields needed to build GLM4-MoE layers.

    Reads from ModelConfig (C++ pybind) or HF dict, with config_json used to
    resolve fields that the legacy loader overwrites on model_config.
    """

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    hidden_size = _get(model_config, "hidden_size", 4096)
    num_layers = _get(
        model_config, "num_layers", _get(model_config, "num_hidden_layers", 28)
    )
    vocab_size = _get(model_config, "vocab_size", 151936)

    attn_config = _get(model_config, "attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 32)
        num_kv_heads = _get(attn_config, "kv_head_num", num_heads)
        head_dim = _get(attn_config, "size_per_head", hidden_size // num_heads)
    else:
        num_heads = _get(model_config, "num_attention_heads", 32)
        num_kv_heads = _get(model_config, "num_key_value_heads", num_heads)
        head_dim = _get(model_config, "head_dim", hidden_size // num_heads)

    rms_norm_eps = _get(
        model_config,
        "layernorm_eps",
        _get(model_config, "rms_norm_eps", 1e-6),
    )

    # Dense FFN width — the legacy loader overwrites inter_size to
    # n_shared_experts * moe_intermediate_size, so read from config_json first.
    dense_intermediate_size = None
    if config_json:
        dense_intermediate_size = config_json.get("intermediate_size")
    if dense_intermediate_size is None:
        dense_intermediate_size = _get(
            model_config,
            "intermediate_size",
            _get(model_config, "inter_size", 13696),
        )

    # MoE-specific fields
    num_experts = _get(
        model_config, "expert_num", _get(model_config, "n_routed_experts", 0)
    )
    top_k = _get(model_config, "moe_k", _get(model_config, "num_experts_per_tok", 0))
    moe_intermediate_size = _get(
        model_config,
        "moe_inter_size",
        _get(model_config, "moe_intermediate_size", 0),
    )
    if config_json:
        if not moe_intermediate_size:
            moe_intermediate_size = config_json.get("moe_intermediate_size", 0)
        if not num_experts:
            num_experts = config_json.get("n_routed_experts", 0)
        if not top_k:
            top_k = config_json.get("num_experts_per_tok", 0)
    if num_experts <= 0 or top_k <= 0 or moe_intermediate_size <= 0:
        raise ValueError(
            f"GLM4-MoE config missing fields: expert_num={num_experts}, "
            f"moe_k={top_k}, moe_inter_size={moe_intermediate_size}"
        )

    # Shared expert
    n_shared_experts = _get(model_config, "n_shared_experts", 1)
    if config_json:
        n_shared_experts = config_json.get("n_shared_experts", n_shared_experts)
    shared_expert_intermediate_size = n_shared_experts * moe_intermediate_size

    # MoE layer index — prefer model_config.moe_layer_index (set by legacy
    # loader), otherwise compute from first_k_dense_replace in config.json.
    moe_layer_index_raw = _get(model_config, "moe_layer_index", None)
    if moe_layer_index_raw is not None and isinstance(
        moe_layer_index_raw, (list, tuple)
    ):
        moe_layer_index = list(moe_layer_index_raw)
    else:
        first_k_dense_replace = None
        if config_json:
            first_k_dense_replace = config_json.get("first_k_dense_replace")
        if first_k_dense_replace is None:
            first_k_dense_replace = _get(model_config, "first_k_dense_replace", 0)
        moe_layer_index = [i for i in range(num_layers) if i >= first_k_dense_replace]

    # QK norm
    use_qk_norm = _get(model_config, "qk_norm", False)
    if config_json:
        use_qk_norm = config_json.get("use_qk_norm", use_qk_norm)

    # Routing params
    scoring_func = _get(model_config, "scoring_func", 1)  # 0=softmax, 1=sigmoid
    routed_scaling_factor = _get(model_config, "routed_scaling_factor", 1.0)
    n_group = _get(model_config, "moe_n_group", _get(model_config, "n_group", 1))
    topk_group = _get(
        model_config, "moe_topk_group", _get(model_config, "topk_group", 1)
    )
    has_moe_norm = _get(
        model_config,
        "has_moe_norm",
        _get(model_config, "norm_topk_prob", False),
    )
    if config_json:
        n_group = config_json.get("n_group", n_group)
        topk_group = config_json.get("topk_group", topk_group)
        has_moe_norm = config_json.get("norm_topk_prob", has_moe_norm)
        routed_scaling_factor = config_json.get(
            "routed_scaling_factor", routed_scaling_factor
        )

    # Correction bias — detect from topk_method or scoring_func in config.json.
    # GLM-4.5 models use sigmoid scoring with correction bias but do NOT set
    # topk_method or scoring_func in their config.json. The old loader detects
    # this by probing weight keys directly. For the new loader, we default to
    # True for glm4_moe (extra param is harmless if weight is absent) and also
    # check topk_method/scoring_func for other model types sharing this code.
    has_e_score_correction = _get(model_config, "has_e_score_correction", False)
    if not has_e_score_correction and config_json:
        topk_method = config_json.get("topk_method", "")
        scoring_func_str = config_json.get("scoring_func", "")
        model_type_str = config_json.get("model_type", "")
        has_e_score_correction = (
            topk_method == "noaux_tc"
            or scoring_func_str == "sigmoid"
            or model_type_str == "glm4_moe"
        )

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
        moe_layer_index=moe_layer_index,
        use_qk_norm=use_qk_norm,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        n_group=n_group,
        topk_group=topk_group,
        has_moe_norm=has_moe_norm,
        has_e_score_correction=has_e_score_correction,
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


class Glm4MoeForCausalLM(GptModelBase):
    """GLM4-MoE for the new weight loader.

    WEIGHTS_MAPPER strips the ``model.`` prefix. All submodule names match
    HF ckpt keys directly, so RtpModule.load_weights can dispatch weights
    without any fusion-time mapping.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        if isinstance(weights, dict):
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
            logging.info(
                "[Glm4MoeForCausalLM] lm_head.weight not found in ckpt; "
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

        # Resolve ckpt_path from model_config to read config.json — fields
        # like intermediate_size (dense FFN) are overwritten on model_config
        # by the legacy loader.
        ckpt_path = ""
        if hasattr(model_config, "ckpt_path") and model_config.ckpt_path:
            ckpt_path = model_config.ckpt_path
        config_json = _read_config_json(ckpt_path)

        cfg = _extract_glm4_moe_config_values(model_config, load_config, config_json)

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        moe_layer_set = set(cfg["moe_layer_index"])
        self.layers = nn.ModuleList()
        for i in range(cfg["num_layers"]):
            is_moe = i in moe_layer_set
            layer = Glm4MoeDecoderLayer(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                layer_idx=i,
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                rms_norm_eps=cfg["rms_norm_eps"],
                use_qk_norm=cfg["use_qk_norm"],
                is_moe_layer=is_moe,
                dense_intermediate_size=cfg["dense_intermediate_size"],
                moe_intermediate_size=cfg["moe_intermediate_size"],
                shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                model_config=cfg["model_config"],
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                scoring_func=cfg["scoring_func"],
                routed_scaling_factor=cfg["routed_scaling_factor"],
                n_group=cfg["n_group"],
                topk_group=cfg["topk_group"],
                has_moe_norm=cfg["has_moe_norm"],
                correction_bias=cfg["has_e_score_correction"],
            )
            self.layers.append(layer)

        self.norm = RMSNorm(
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
        hidden_states = self.embed_tokens(input_ids)
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
