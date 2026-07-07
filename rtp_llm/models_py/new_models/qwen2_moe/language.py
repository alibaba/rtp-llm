"""Qwen2-MoE for the new loader.

Supports BF16 / FP8 ckpts with TP and EP parallelism. Reuses dense Qwen2's
attention block and the loader-agnostic FusedMoeFactory.

Differs from Qwen3-MoE by having a shared expert alongside the routed experts:
  * Routed experts: BaseMoEExperts (reuse FP8 support).
  * Shared expert: gated FFN (gate_proj + up_proj -> down_proj) that is always
    activated, scaled by a shared_expert_gate sigmoid.

Per-expert ckpt streaming is handled inside BaseMoEExperts.load_weights. EP
support is provided by NewModelLoader._apply_ep_filter and BaseMoEExperts
remaps global expert IDs to local IDs internally.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.activation import silu_and_mul
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
from rtp_llm.models_py.modules import SelectTopk
from rtp_llm.models_py.new_models.qwen2_vl.language import Qwen2Attention
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs


class Qwen2SharedExpert(RtpModule):
    """Gated shared expert FFN (always activated).

    Mirrors the dense Qwen2 MLP layout but does NOT all-reduce inside forward:
    the surrounding Qwen2MoeBlock / Qwen2MoeDecoderLayer perform a single
    all-reduce after combining routed-expert and shared-expert outputs.
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
        x = silu_and_mul(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2MoeBlock(RtpModule):
    """Routed-expert MoE block with shared expert (Qwen2-MoE)."""

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
    ):
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k

        # Router: hidden -> num_experts. NOT TP-sharded.
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_experts,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)
        self.experts = BaseMoEExperts(
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
        self.shared_expert = Qwen2SharedExpert(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        # Shared expert gate: hidden -> 1, scaled with sigmoid. NOT TP-sharded.
        self.shared_expert_gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=1,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="shared_expert_gate",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=(
                self.experts.fused_moe.topk_ids_dtype
                if self.experts.fused_moe is not None
                else torch.int32
            ),
            device=hidden_states.device,
        )
        self.select_topk(router_logits_fp32, topk_ids, topk_weights)
        routed = self.experts(hidden_states, topk_weights, topk_ids)

        shared = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states).float())
        shared = shared_gate.to(hidden_states.dtype) * shared

        if self.ep_size > 1:
            # EP mode: routed experts output is already complete (EP combine
            # inside FusedMoe), but the shared expert is still TP-partial.
            # Reduce the shared branch before adding.
            shared = all_reduce(shared, group=Group.TP)

        return routed + shared


class Qwen2MoeDecoderLayer(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        head_dim: int,
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
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        # Qwen2 attention: standard MHA with qkv bias, no q/k norm.
        self.self_attn = Qwen2Attention(
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
        self.mlp = Qwen2MoeBlock(
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
        if self.mlp.ep_size <= 1 and self.mlp.tp_size > 1:
            # TP-only mode: MoE FFN inner-dim is TP-sharded; reduce across TP ranks.
            # EP mode: FusedMoe executor handles EP combine internally.
            hidden_states = all_reduce(hidden_states, group=Group.TP)
        hidden_states = residual + hidden_states
        return hidden_states


def _extract_moe_config_values(model_config: Any, load_config: Any) -> Dict[str, Any]:
    """Pull all fields needed to build Qwen2MoE layers."""

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    hidden_size = _get(model_config, "hidden_size", 2048)
    num_layers = _get(
        model_config, "num_layers", _get(model_config, "num_hidden_layers", 48)
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

    # MoE-specific fields. ModelConfig uses legacy names set by the old loader;
    # HF dict uses the original HuggingFace names.
    num_experts = _get(model_config, "expert_num", _get(model_config, "num_experts", 0))
    top_k = _get(model_config, "moe_k", _get(model_config, "num_experts_per_tok", 0))
    moe_intermediate_size = _get(
        model_config,
        "moe_inter_size",
        _get(model_config, "moe_intermediate_size", 0),
    )
    shared_expert_intermediate_size = _get(
        model_config,
        "inter_size",
        _get(model_config, "shared_expert_intermediate_size", 0),
    )
    if num_experts <= 0 or top_k <= 0 or moe_intermediate_size <= 0:
        raise ValueError(
            f"Qwen2-MoE config missing fields: expert_num={num_experts}, "
            f"moe_k={top_k}, moe_inter_size={moe_intermediate_size}"
        )
    if shared_expert_intermediate_size <= 0:
        raise ValueError(
            f"Qwen2-MoE config missing shared_expert_intermediate_size: "
            f"inter_size={shared_expert_intermediate_size}"
        )

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
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=num_layers,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
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


class Qwen2MoeForCausalLM(GptModelBase):

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        import logging

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
                "[Qwen2MoeForCausalLM] lm_head.weight not found in ckpt; "
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

        cfg = _extract_moe_config_values(model_config, load_config)

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    moe_intermediate_size=cfg["moe_intermediate_size"],
                    shared_expert_intermediate_size=cfg[
                        "shared_expert_intermediate_size"
                    ],
                    num_experts=cfg["num_experts"],
                    top_k=cfg["top_k"],
                    head_dim=cfg["head_dim"],
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
