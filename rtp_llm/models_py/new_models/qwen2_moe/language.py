import logging
import math
from numbers import Real
from typing import Any, Optional

import torch
import torch.nn as nn
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import FakeBalanceExpert, SelectTopk
from rtp_llm.models_py.new_models.model_base import (
    required_config_value,
    select_block_map_for_layer,
)
from rtp_llm.models_py.new_models.qwen2.language import (
    Qwen2Attention,
    _extract_config_values,
    _positive_int,
    _validate_supported_parallelism,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs


class Qwen2Experts(BaseMoEExperts):
    """Qwen2 expert layout implemented by the shared newloader MoE layer."""


class Qwen2SharedExpert(RtpModule):
    """Tensor-parallel gated FFN used by every token."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        prefix: str,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
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
            prefix=f"{prefix}.down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(hidden_states)))


class Qwen2MoeBlock(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        ffn_tp_size: int,
        ffn_tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: ModelConfig,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        prefix: str,
    ):
        super().__init__()
        self.top_k = top_k
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_experts,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix=f"{prefix}.gate",
            bias=False,
            params_dtype=params_dtype,
        )
        if model_config.expert_num != num_experts or model_config.moe_k != top_k:
            raise ValueError(
                "Qwen2 MoE router dimensions disagree with ModelConfig: "
                f"experts={model_config.expert_num}/{num_experts}, "
                f"top_k={model_config.moe_k}/{top_k}"
            )
        self.select_topk = SelectTopk(config=model_config)
        fake_balance_expert = getattr(moe_config, "fake_balance_expert", False)
        if not isinstance(fake_balance_expert, bool):
            raise TypeError("moe_config.fake_balance_expert must be a bool")
        if fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=num_experts,
                moe_k=top_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=ep_size,
            )
        else:
            self.fake_balance_expert = None
        self.experts = Qwen2Experts(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            tp_size=ffn_tp_size,
            tp_rank=ffn_tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=params_dtype,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.experts",
        )
        self.shared_expert = Qwen2SharedExpert(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            tp_size=ffn_tp_size,
            tp_rank=ffn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=1,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.experts.fused_moe is None:
            raise RuntimeError(
                "Qwen2 MoE executor is not initialized; load and postprocess "
                "all expert weights before forward"
            )
        if hidden_states.dim() != 2:
            raise ValueError(
                "Qwen2 MoE expects a two-dimensional token matrix, got "
                f"{tuple(hidden_states.shape)}"
            )
        router_logits = self.gate(hidden_states).float()
        shape = (hidden_states.shape[0], self.top_k)
        topk_weights = torch.empty(
            shape, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            shape,
            dtype=self.experts.fused_moe.topk_ids_dtype,
            device=hidden_states.device,
        )
        self.select_topk(router_logits, topk_ids, topk_weights)
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)
        routed = self.experts(hidden_states, topk_weights, topk_ids)
        shared = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states).float())
        return routed + shared * shared_gate.to(dtype=shared.dtype)


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
        attn_tp_size: int,
        attn_tp_rank: int,
        ffn_tp_size: int,
        ffn_tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: ModelConfig,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        rms_norm_eps: float,
        prefix: str,
    ):
        super().__init__()
        self.input_layernorm = RMSResNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=attn_tp_size,
            tp_rank=attn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSResNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.mlp = Qwen2MoeBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            layer_idx=layer_idx,
            ffn_tp_size=ffn_tp_size,
            ffn_tp_rank=ffn_tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _model_value(model_config: Any, *names: str):
    return required_config_value(model_config, *names)


def _extract_moe_config_values(model_config: ModelConfig, load_config: Any):
    values = _extract_config_values(
        model_config, load_config, validate_intermediate_size=False
    )
    moe_style = _model_value(model_config, "moe_style")
    if isinstance(moe_style, bool) or not isinstance(moe_style, int) or moe_style != 2:
        raise ValueError(
            "Qwen2 MoE newloader requires moe_style=2 (shared and routed "
            f"experts), got {moe_style!r}"
        )
    num_experts = _model_value(model_config, "expert_num")
    top_k = _model_value(model_config, "moe_k")
    moe_intermediate_size = _model_value(model_config, "moe_inter_size")
    shared_expert_intermediate_size = _model_value(model_config, "inter_size")
    num_experts = _positive_int(num_experts, "num_experts")
    top_k = _positive_int(top_k, "top_k")
    moe_intermediate_size = _positive_int(
        moe_intermediate_size, "moe_intermediate_size"
    )
    shared_expert_intermediate_size = _positive_int(
        shared_expert_intermediate_size, "shared_expert_intermediate_size"
    )
    expected_moe_layers = list(range(values["num_layers"]))
    moe_layer_index = getattr(model_config, "moe_layer_index", None)
    if (
        not isinstance(moe_layer_index, (list, tuple))
        or list(moe_layer_index) != expected_moe_layers
    ):
        raise ValueError(
            "Qwen2 MoE newloader requires every decoder layer to be an MoE "
            f"layer; expected moe_layer_index={expected_moe_layers}, got "
            f"{moe_layer_index!r}"
        )
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} cannot exceed num_experts={num_experts}")
    ep_size = _positive_int(required_config_value(load_config, "ep_size"), "EP size")
    ep_rank = required_config_value(load_config, "ep_rank")
    if (
        isinstance(ep_rank, bool)
        or not isinstance(ep_rank, int)
        or not 0 <= ep_rank < ep_size
    ):
        raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
        )
    parallelism_config = required_config_value(load_config, "parallelism_config")
    moe_config = required_config_value(load_config, "moe_config")
    dp_size = getattr(parallelism_config, "dp_size", None)
    dp_rank = getattr(parallelism_config, "dp_rank", None)
    if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
        raise ValueError(f"DP size must be a positive integer, got {dp_size!r}")
    if (
        isinstance(dp_rank, bool)
        or not isinstance(dp_rank, int)
        or not 0 <= dp_rank < dp_size
    ):
        raise ValueError(f"Invalid DP partition: rank={dp_rank}, size={dp_size}")
    eplb_config = getattr(model_config, "eplb_config", None)
    enable_eplb = getattr(eplb_config, "enable_eplb", False)
    if callable(enable_eplb):
        enable_eplb = enable_eplb()
    if not isinstance(enable_eplb, bool):
        raise TypeError("eplb_config.enable_eplb must be a bool")
    if enable_eplb:
        raise ValueError("EPLB is not supported by the Qwen2 MoE newloader path")
    values.update(
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        ep_size=ep_size,
        ep_rank=ep_rank,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )
    return values


class Qwen2MoeForCausalLM(GptModelBase):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        has_lm_head = False

        def track(mapped):
            nonlocal has_lm_head
            for name, tensor in mapped:
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        super().load_weights(track(self.WEIGHTS_MAPPER.apply(iterator)))
        if not has_lm_head and self.tie_word_embeddings:
            logging.info("Qwen2MoeForCausalLM is tying lm_head to embed_tokens")
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def process_weights_after_loading(self) -> None:
        if self._lm_head_postprocessed:
            return
        processed = self.lm_head.weight
        if self.normalize_lm_head_weight:
            processed = torch.nn.functional.normalize(processed, dim=1)
        if self.logit_scale != 1.0:
            processed = processed * self.logit_scale
        if processed is not self.lm_head.weight:
            with torch.no_grad():
                self.lm_head.weight.copy_(processed)
        self._lm_head_postprocessed = True

    def __init__(self, model_config: ModelConfig, load_config: Any):
        if not isinstance(model_config, ModelConfig):
            raise TypeError(
                "Qwen2 MoE newloader requires a typed ModelConfig; normalize "
                "raw Hugging Face config.json data at the BaseModel boundary"
            )
        parallelism_config = required_config_value(load_config, "parallelism_config")
        _validate_supported_parallelism(parallelism_config)
        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=getattr(load_config, "fmha_config", None),
            device_resource_config=getattr(load_config, "device_resource_config", None),
        )
        cfg = _extract_moe_config_values(model_config, load_config)
        tie_word_embeddings = getattr(model_config, "tie_word_embeddings", False)
        if not isinstance(tie_word_embeddings, bool):
            raise TypeError("tie_word_embeddings must be a bool")
        self.tie_word_embeddings = tie_word_embeddings
        normalize = getattr(model_config, "normalize_lm_head_weight", False)
        if not isinstance(normalize, bool):
            raise TypeError("normalize_lm_head_weight must be a bool")
        raw_scale = getattr(model_config, "logit_scale", 1.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, Real):
            raise TypeError("logit_scale must be a finite real number")
        self.logit_scale = float(raw_scale)
        if not math.isfinite(self.logit_scale):
            raise ValueError("logit_scale must be finite")
        self.normalize_lm_head_weight = normalize
        self._lm_head_postprocessed = False

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
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
                    layer_idx=layer_idx,
                    attn_tp_size=cfg["attn_tp_size"],
                    attn_tp_rank=cfg["attn_tp_rank"],
                    ffn_tp_size=cfg["ffn_tp_size"],
                    ffn_tp_rank=cfg["ffn_tp_rank"],
                    ep_size=cfg["ep_size"],
                    ep_rank=cfg["ep_rank"],
                    model_config=cfg["model_config"],
                    parallelism_config=cfg["parallelism_config"],
                    moe_config=cfg["moe_config"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                    prefix=f"layers.{layer_idx}",
                )
                for layer_idx in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def runtime_weight_view(self) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embed_tokens.weight,
            "final_layernorm.gamma": self.norm.weight,
            "lm_head": self.lm_head.weight,
        }

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.embed_tokens(inputs.input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, layer_idx)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=(
                    self.kv_cache.get_layer_cache(layer_idx) if self.kv_cache else None
                ),
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
