"""DeepSeek-VL2 language runtime for the streaming newloader.

The VL2 family contains two language-attention layouts:

* ``deepseek-vl2-tiny`` uses ordinary, bias-free Q/K/V attention.
* ``deepseek-vl2-small`` and ``deepseek-vl2`` use DeepSeek MLA with a
  direct query projection (no Q-LoRA).

All variants use the same DeepSeekMoE feed-forward layout and checkpoint
prefixes under ``language.model``.
"""

import json
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.new_models.deepseek_v3.attention import DeepSeekV32MlaAttention
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    MlaRuntimeLayoutMixin,
    _bool_value,
    _partition,
    _positive_float,
    build_rope_cache,
    nonnegative_int,
    positive_int,
    validate_deepseek_mla_backend,
    validate_deepseek_newloader_eplb,
)
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3.moe import (
    DeepSeekV32MoEBlock,
    normalize_topk_method,
)
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs

logger = logging.getLogger(__name__)


def _read_language_config(ckpt_path: str) -> dict[str, Any]:
    if not ckpt_path:
        raise ValueError("DeepSeek-VL2 newloader requires a checkpoint path")
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"DeepSeek-VL2 newloader requires {config_path!r}")
    with open(config_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    language_config = payload.get("language_config")
    if not isinstance(language_config, dict):
        raise ValueError(
            "DeepSeek-VL2 config.json must contain a language_config object"
        )
    return language_config


def _resolve_scoring_func(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "softmax":
            return 0
        if normalized == "sigmoid":
            return 1
        raise ValueError(f"unsupported DeepSeek-VL2 scoring_func={value!r}")
    value = nonnegative_int(value, "scoring_func")
    if value not in (0, 1):
        raise ValueError(f"unsupported DeepSeek-VL2 scoring_func={value}")
    return value


def _extract_config_values(
    model_config: Any,
    load_config: Any,
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_config, Mapping):
        raise TypeError("language_config must be a mapping")

    validate_deepseek_newloader_eplb(model_config, "DeepSeek-VL2")

    hidden_size = positive_int(model_config.hidden_size, "hidden_size")
    checkpoint_hidden_size = raw_config.get("hidden_size")
    if (
        checkpoint_hidden_size is not None
        and positive_int(checkpoint_hidden_size, "hidden_size") != hidden_size
    ):
        raise ValueError(
            f"hidden_size mismatch: ModelConfig({hidden_size})/"
            f"checkpoint({checkpoint_hidden_size})"
        )

    num_layers = positive_int(model_config.num_layers, "num_layers")
    checkpoint_num_layers = positive_int(
        raw_config.get("num_hidden_layers", num_layers),
        "num_hidden_layers",
    )
    if num_layers > checkpoint_num_layers:
        raise ValueError(
            f"num_layers={num_layers} exceeds checkpoint "
            f"num_hidden_layers={checkpoint_num_layers}"
        )

    vocab_size = positive_int(model_config.vocab_size, "vocab_size")
    checkpoint_vocab_size = raw_config.get("vocab_size")
    if (
        checkpoint_vocab_size is not None
        and positive_int(checkpoint_vocab_size, "vocab_size") != vocab_size
    ):
        raise ValueError(
            f"vocab_size mismatch: ModelConfig({vocab_size})/"
            f"checkpoint({checkpoint_vocab_size})"
        )

    max_seq_len = positive_int(model_config.max_seq_len, "max_seq_len")
    attn_config = model_config.attn_config
    num_heads = positive_int(attn_config.head_num, "num_attention_heads")
    checkpoint_num_heads = raw_config.get("num_attention_heads")
    if (
        checkpoint_num_heads is not None
        and positive_int(checkpoint_num_heads, "num_attention_heads") != num_heads
    ):
        raise ValueError(
            f"num_attention_heads mismatch: ModelConfig({num_heads})/"
            f"checkpoint({checkpoint_num_heads})"
        )
    num_kv_heads = positive_int(attn_config.kv_head_num, "num_key_value_heads")
    checkpoint_num_kv_heads = raw_config.get("num_key_value_heads")
    if (
        checkpoint_num_kv_heads is not None
        and positive_int(checkpoint_num_kv_heads, "num_key_value_heads") != num_kv_heads
    ):
        raise ValueError(
            f"num_key_value_heads mismatch: ModelConfig({num_kv_heads})/"
            f"checkpoint({checkpoint_num_kv_heads})"
        )
    if hidden_size % num_heads:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by "
            f"num_attention_heads={num_heads}"
        )
    head_dim = hidden_size // num_heads

    checkpoint_use_mla = _bool_value(raw_config.get("use_mla", True), "use_mla")
    use_mla = _bool_value(attn_config.use_mla, "attn_config.use_mla")
    if checkpoint_use_mla != use_mla:
        raise ValueError(
            f"use_mla mismatch: ModelConfig({use_mla})/"
            f"checkpoint({checkpoint_use_mla})"
        )
    validate_deepseek_mla_backend(model_config, use_mla, "DeepSeek-VL2")
    q_lora_rank_value = raw_config.get("q_lora_rank", 1536)
    q_lora_rank = (
        0
        if q_lora_rank_value is None
        else nonnegative_int(q_lora_rank_value, "q_lora_rank")
    )
    kv_lora_rank_value = raw_config.get("kv_lora_rank", 512)
    kv_lora_rank = (
        0
        if kv_lora_rank_value is None
        else nonnegative_int(kv_lora_rank_value, "kv_lora_rank")
    )
    nope_head_dim = nonnegative_int(
        raw_config.get("qk_nope_head_dim", 128), "qk_nope_head_dim"
    )
    rope_head_dim = nonnegative_int(
        raw_config.get("qk_rope_head_dim", 64), "qk_rope_head_dim"
    )
    v_head_dim = nonnegative_int(raw_config.get("v_head_dim", 128), "v_head_dim")
    if use_mla:
        mla_dimensions = (
            ("q_lora_rank", q_lora_rank, attn_config.q_lora_rank),
            ("kv_lora_rank", kv_lora_rank, attn_config.kv_lora_rank),
            ("qk_nope_head_dim", nope_head_dim, attn_config.nope_head_dim),
            ("qk_rope_head_dim", rope_head_dim, attn_config.rope_head_dim),
            ("v_head_dim", v_head_dim, attn_config.v_head_dim),
        )
        for name, checkpoint_value, model_value in mla_dimensions:
            normalized_model_value = nonnegative_int(model_value, f"ModelConfig {name}")
            if normalized_model_value != checkpoint_value:
                raise ValueError(
                    f"{name} mismatch: ModelConfig({normalized_model_value})/"
                    f"checkpoint({checkpoint_value})"
                )
        if kv_lora_rank == 0:
            raise ValueError("MLA requires kv_lora_rank > 0")
        if nope_head_dim == 0 or rope_head_dim == 0 or v_head_dim == 0:
            raise ValueError(
                "MLA requires positive qk_nope_head_dim, qk_rope_head_dim, "
                "and v_head_dim"
            )
        if rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")

    dense_intermediate_size = positive_int(
        raw_config.get("intermediate_size", 11008), "intermediate_size"
    )
    moe_intermediate_size = positive_int(
        raw_config.get("moe_intermediate_size", 1407),
        "moe_intermediate_size",
    )
    num_experts = positive_int(
        raw_config.get("n_routed_experts"),
        "n_routed_experts",
    )
    top_k = positive_int(
        raw_config.get("num_experts_per_tok"),
        "num_experts_per_tok",
    )
    if top_k > num_experts:
        raise ValueError(
            f"num_experts_per_tok={top_k} exceeds n_routed_experts={num_experts}"
        )
    n_shared_experts = positive_int(
        raw_config.get("n_shared_experts", 1), "n_shared_experts"
    )
    shared_expert_intermediate_size = n_shared_experts * moe_intermediate_size
    first_k_dense_replace = nonnegative_int(
        raw_config.get("first_k_dense_replace", 0), "first_k_dense_replace"
    )
    moe_layer_freq = positive_int(raw_config.get("moe_layer_freq", 1), "moe_layer_freq")
    moe_layer_index = [
        layer_idx
        for layer_idx in range(num_layers)
        if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
    ]

    scoring_func = _resolve_scoring_func(raw_config.get("scoring_func", 0))
    routed_scaling_factor = _positive_float(
        raw_config.get("routed_scaling_factor", 1.0),
        "routed_scaling_factor",
    )
    n_group = positive_int(raw_config.get("n_group", 1), "n_group")
    topk_group = positive_int(raw_config.get("topk_group", 1), "topk_group")
    if topk_group > n_group:
        raise ValueError(f"topk_group={topk_group} exceeds n_group={n_group}")
    has_moe_norm = _bool_value(
        raw_config.get("norm_topk_prob", False), "norm_topk_prob"
    )
    topk_method = normalize_topk_method(raw_config.get("topk_method", "greedy")).value
    correction_bias = topk_method == "noaux_tc"
    if moe_layer_index and topk_method in {"group_limited_greedy", "noaux_tc"}:
        if num_experts % n_group:
            raise ValueError(
                f"n_routed_experts={num_experts} must be divisible by "
                f"n_group={n_group}"
            )
        capacity = topk_group * (num_experts // n_group)
        if top_k > capacity:
            raise ValueError(
                f"num_experts_per_tok={top_k} exceeds grouped capacity={capacity}"
            )

    rms_norm_eps = _positive_float(model_config.layernorm_eps, "rms_norm_eps")
    attn_tp_size, attn_tp_rank = _partition(load_config, "attn_tp")
    ffn_tp_size, ffn_tp_rank = _partition(load_config, "ffn_tp")
    lm_head_tp_size, lm_head_tp_rank = _partition(load_config, "lm_head_tp")
    ep_size, ep_rank = _partition(load_config, "ep")
    if num_heads % attn_tp_size:
        raise ValueError(
            f"num_attention_heads={num_heads} must be divisible by "
            f"attn_tp_size={attn_tp_size}"
        )
    if num_experts % ep_size:
        raise ValueError(
            f"n_routed_experts={num_experts} must be divisible by ep_size={ep_size}"
        )
    if use_mla and num_kv_heads != num_heads:
        raise ValueError(
            "DeepSeek-VL2 MLA expects num_key_value_heads to equal "
            "num_attention_heads"
        )

    params_dtype = load_config.compute_dtype
    if not isinstance(params_dtype, torch.dtype):
        raise TypeError(f"compute_dtype must be torch.dtype, got {params_dtype!r}")
    model_tie_word_embeddings = _bool_value(
        model_config.tie_word_embeddings,
        "model_config.tie_word_embeddings",
    )
    checkpoint_tie_word_embeddings = _bool_value(
        raw_config.get("tie_word_embeddings", False),
        "language_config.tie_word_embeddings",
    )
    tie_word_embeddings = model_tie_word_embeddings or checkpoint_tie_word_embeddings
    if tie_word_embeddings and (
        attn_tp_size != lm_head_tp_size or attn_tp_rank != lm_head_tp_rank
    ):
        raise ValueError(
            "tied DeepSeek-VL2 embeddings require matching attention and "
            "LM-head TP partitions"
        )
    enable_fp32_lm_head = model_config.enable_fp32_lm_head
    enable_fp32_lm_head = _bool_value(enable_fp32_lm_head, "enable_fp32_lm_head")

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "checkpoint_num_layers": checkpoint_num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "use_mla": use_mla,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "nope_head_dim": nope_head_dim,
        "rope_head_dim": rope_head_dim,
        "v_head_dim": v_head_dim,
        "dense_intermediate_size": dense_intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
        "moe_layer_index": moe_layer_index,
        "scoring_func": scoring_func,
        "routed_scaling_factor": routed_scaling_factor,
        "n_group": n_group,
        "topk_group": topk_group,
        "topk_method": topk_method,
        "has_moe_norm": has_moe_norm,
        "correction_bias": correction_bias,
        "rms_norm_eps": rms_norm_eps,
        "attn_tp_size": attn_tp_size,
        "attn_tp_rank": attn_tp_rank,
        "ffn_tp_size": ffn_tp_size,
        "ffn_tp_rank": ffn_tp_rank,
        "lm_head_tp_size": lm_head_tp_size,
        "lm_head_tp_rank": lm_head_tp_rank,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "quant_config": load_config.quant_config,
        "params_dtype": params_dtype,
        "lm_head_params_dtype": (
            torch.float32 if enable_fp32_lm_head else params_dtype
        ),
        "tie_word_embeddings": tie_word_embeddings,
        "model_config": model_config,
        "parallelism_config": load_config.parallelism_config,
        "moe_config": load_config.moe_config,
    }


class DeepSeekVLV2Attention(RtpModule):
    """Bias-free standard attention used by the tiny VL2 checkpoint."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        prefix: str = "self_attn",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
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
        output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        output = output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output)


class DeepSeekVLV2DecoderLayer(RtpModule):
    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        layer_idx: int,
        is_moe_layer: bool,
    ) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}"
        self.input_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        if cfg["use_mla"]:
            self.self_attn = DeepSeekV32MlaAttention(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                q_lora_rank=cfg["q_lora_rank"],
                kv_lora_rank=cfg["kv_lora_rank"],
                nope_head_dim=cfg["nope_head_dim"],
                rope_head_dim=cfg["rope_head_dim"],
                v_head_dim=cfg["v_head_dim"],
                layer_idx=layer_idx,
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                layernorm_eps=cfg["rms_norm_eps"],
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = DeepSeekVLV2Attention(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                layer_idx=layer_idx,
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                prefix=f"{prefix}.self_attn",
            )
        self.post_attention_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        if is_moe_layer:
            self.mlp = DeepSeekV32MoEBlock(
                hidden_size=cfg["hidden_size"],
                moe_intermediate_size=cfg["moe_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                layer_idx=layer_idx,
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                model_config=cfg["model_config"],
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                has_shared_expert=True,
                shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
                scoring_func=cfg["scoring_func"],
                routed_scaling_factor=cfg["routed_scaling_factor"],
                n_group=cfg["n_group"],
                topk_group=cfg["topk_group"],
                topk_method=cfg["topk_method"],
                has_moe_norm=cfg["has_moe_norm"],
                correction_bias=cfg["correction_bias"],
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepSeekV32MLP(
                hidden_size=cfg["hidden_size"],
                intermediate_size=cfg["dense_intermediate_size"],
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
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


class DeepSeekVLV2ForCausalLM(MlaRuntimeLayoutMixin, GptModelBase):
    """Language backbone with strict VL2 checkpoint filtering and injection."""

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "language.model.": "",
            "language.lm_head.": "lm_head.",
        }
    )

    def __init__(self, model_config: Any, load_config: Any) -> None:
        parallelism_config = load_config.parallelism_config
        if parallelism_config is None:
            raise ValueError("DeepSeek-VL2 newloader requires parallelism_config")
        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=load_config.fmha_config,
            device_resource_config=load_config.device_resource_config,
        )
        ckpt_path = model_config.ckpt_path
        raw_config = _read_language_config(ckpt_path)
        cfg = _extract_config_values(model_config, load_config, raw_config)
        self.use_mla = cfg["use_mla"]
        self.tie_word_embeddings = cfg["tie_word_embeddings"]
        self._checkpoint_num_layers = cfg["checkpoint_num_layers"]
        self._keep_mla_checkpoint_weights = load_config.keep_mla_checkpoint_weights

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        moe_layers = set(cfg["moe_layer_index"])
        self.layers = nn.ModuleList(
            [
                DeepSeekVLV2DecoderLayer(
                    cfg=cfg,
                    layer_idx=layer_idx,
                    is_moe_layer=layer_idx in moe_layers,
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
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

        if self.use_mla:
            rope_config = dict(raw_config)
            rope_config.setdefault("qk_rope_head_dim", cfg["rope_head_dim"])
            device = torch.device(load_config.device)
            cos_sin_cache = build_rope_cache(
                rope_config,
                cfg["max_seq_len"],
                device,
            )
            self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
        else:
            self.register_buffer("cos_sin_cache", None, persistent=False)

        self._mla_kernel_layout = None

    @staticmethod
    def _checkpoint_layer_index(name: str) -> Optional[int]:
        prefix = "language.model.layers."
        if not name.startswith(prefix):
            return None
        layer_text, separator, _ = name[len(prefix) :].partition(".")
        if not separator or not layer_text.isdigit():
            return None
        return int(layer_text)

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        # The language loader owns every non-vision tensor. This both partitions
        # the multimodal checkpoint and ensures an unknown top-level tensor
        # reaches the strict dispatcher instead of being ignored by both
        # component loaders.
        num_layers = len(self.layers)

        def should_load(name: str) -> bool:
            if name.startswith(("vision.", "projector.")) or name in {
                "image_newline",
                "view_seperator",
            }:
                return False
            layer_idx = self._checkpoint_layer_index(name)
            if layer_idx is None:
                return True
            # A ModelConfig layer override intentionally truncates the model.
            # Skip only layers known to belong to the declared checkpoint;
            # out-of-range or malformed layer keys must still fail strictly.
            if num_layers <= layer_idx < self._checkpoint_num_layers:
                return False
            return True

        return should_load

    def load_weights(self, weights: Any) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        has_lm_head = False

        def track():
            nonlocal has_lm_head
            for name, tensor in iterator:
                if name == "language.lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        super().load_weights(self.WEIGHTS_MAPPER.apply(track()))
        if not has_lm_head and self.tie_word_embeddings:
            logger.info(
                "DeepSeek-VL2 lm_head.weight is absent; tying it to embed_tokens"
            )
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def _embed_inputs(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids = inputs.input_ids
        embedding_inputs = inputs.embedding_inputs
        text_tokens_mask = (
            None if embedding_inputs is None else embedding_inputs.text_tokens_mask
        )
        multimodal_inputs = inputs.multimodal_inputs
        features = (
            []
            if multimodal_inputs is None
            else multimodal_inputs.multimodal_features or []
        )
        locations = (
            None if multimodal_inputs is None else multimodal_inputs.mm_features_locs
        )
        if features and text_tokens_mask is None:
            raise ValueError(
                "DeepSeek-VL2 received multimodal features without " "text_tokens_mask"
            )
        if features and locations is None:
            raise ValueError(
                "DeepSeek-VL2 received multimodal features without " "mm_features_locs"
            )

        if text_tokens_mask is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            if text_tokens_mask.numel() != input_ids.numel():
                raise ValueError(
                    "text_tokens_mask must contain one entry per input token"
                )
            text_mask = text_tokens_mask.to(
                device=input_ids.device, dtype=torch.bool
            ).view_as(input_ids)
            safe_input_ids = torch.where(text_mask, input_ids, 0)
            hidden_states = self.embed_tokens(safe_input_ids)
            hidden_states = hidden_states * text_mask.unsqueeze(-1)

        if multimodal_inputs is None:
            return hidden_states
        return self.multimodal_embedding_injector(hidden_states, features, locations)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if inputs.attention_inputs is None:
            raise ValueError("DeepSeek-VL2 forward requires attention_inputs")
        hidden_states = self._embed_inputs(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            layer_fmha_impl = select_fmha_impl_for_layer(
                fmha_impl, self.kv_cache, layer_idx
            )
            hidden_states, residual = layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=(
                    self.kv_cache.get_layer_cache(layer_idx) if self.kv_cache else None
                ),
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states)


__all__ = [
    "DeepSeekVLV2Attention",
    "DeepSeekVLV2DecoderLayer",
    "DeepSeekVLV2ForCausalLM",
    "_extract_config_values",
]
