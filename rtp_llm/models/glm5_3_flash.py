"""GLM-5.3-Flash model configuration and registration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import DataType, HybridAttentionType


@dataclass(frozen=True)
class Glm53FlashRuntimeConfig:
    """Python-only fields that define GLM-5.3-Flash execution semantics."""

    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    swiglu_limit: float
    dense_intermediate_size: int
    num_shared_experts: int
    kda_gate_lower_bound: float
    kda_gate_lora_rank: int
    index_raw_topk: int
    index_group_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    index_share_for_mtp_iteration: bool
    mla_use_nope: bool


class Glm53FlashModelConfig(ModelConfig):
    _python_fields = ModelConfig._python_fields | {"glm5_3_flash_runtime_config"}
    glm5_3_flash_runtime_config: Glm53FlashRuntimeConfig

    def init_linear_attention_cache_precision(self, kv_cache_config: Any) -> None:
        super().init_linear_attention_cache_precision(kv_cache_config)
        # GLM-5.3-Flash KDA accumulates its recurrent state in FP32. The convolution
        # history continues to follow the model activation dtype.
        self.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32


class Glm53Flash(BaseModel):
    """Text serving path for GLM-5.3-Flash.

    The released checkpoint uses legacy Hugging Face ``Glm5Next*`` identifiers,
    wraps the language model under ``text_config``, and also contains a vision
    tower. RTP-LLM exposes the model as GLM-5.3-Flash and deliberately registers
    the text decoder only; multimodal requests are rejected instead of silently
    placing image embeddings incorrectly.
    """

    @staticmethod
    def _required(config: Dict[str, Any], name: str) -> Any:
        if name not in config:
            raise ValueError(
                f"GLM-5.3-Flash text_config is missing required field {name!r}"
            )
        return config[name]

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be positive, got {parsed}")
        return parsed

    @staticmethod
    def _zero_based_layer_set(
        values: Iterable[Any], name: str, num_layers: int
    ) -> Set[int]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must contain zero-based integer layer ids")
        try:
            raw = list(values)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain zero-based integer layer ids"
            ) from error
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
            raise ValueError(f"{name} must contain zero-based integer layer ids")
        result = set(raw)
        invalid = sorted(value for value in result if value < 0 or value >= num_layers)
        if invalid:
            raise ValueError(
                f"{name} contains out-of-range layers {invalid}; valid range is "
                f"[0, {num_layers})"
            )
        return result

    @classmethod
    def _create_config(cls, ckpt_path: str) -> Glm53FlashModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path, encoding="utf-8") as reader:
            raw = json.load(reader)
        return cls._from_config_json(raw, ckpt_path)

    @classmethod
    def _from_config_json(
        cls, raw: Dict[str, Any], ckpt_path: str = ""
    ) -> Glm53FlashModelConfig:
        if raw.get("model_type") != "glm5_next":
            raise ValueError(
                "expected outer model_type='glm5_next', got "
                f"{raw.get('model_type')!r}"
            )
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "glm5_next_text":
            raise ValueError(
                "GLM-5.3-Flash requires text_config.model_type='glm5_next_text'"
            )

        config = Glm53FlashModelConfig()
        config.ckpt_path = ckpt_path
        config.tokenizer_path = ckpt_path
        config.model_type = "glm5_3_flash"
        cls._parse_basic(raw, text, config)
        cls._parse_attention(text, config)
        cls._parse_moe(text, config)
        cls._parse_hybrid(text, config)
        cls._parse_runtime(text, config)

        # The language path is complete; vision weights remain intentionally
        # unloaded until the GLM-5.3-Flash processor/position-id contract is ported.
        config.mm_model_config.is_multimodal = False
        logging.info(
            "GLM-5.3-Flash config loaded: layers=%d KDA=%d sparse_MLA=%d "
            "experts=%d raw_topk=%d pooled_topk=%d attention_topk=%d",
            config.num_layers,
            sum(
                value == HybridAttentionType.LINEAR
                for value in config.hybrid_attention_config.hybrid_attention_types
            ),
            len(config.attn_config.indexer_layer_ids),
            config.expert_num,
            config.glm5_3_flash_runtime_config.index_raw_topk,
            config.attn_config.indexer_topk,
            config.attn_config.sparse_attention_topk,
        )
        return config

    @classmethod
    def _parse_basic(
        cls,
        outer: Dict[str, Any],
        text: Dict[str, Any],
        config: Glm53FlashModelConfig,
    ) -> None:
        config.num_layers = cls._positive_int(
            cls._required(text, "num_hidden_layers"), "num_hidden_layers"
        )
        config.hidden_size = cls._positive_int(
            cls._required(text, "hidden_size"), "hidden_size"
        )
        config.vocab_size = cls._positive_int(
            cls._required(text, "vocab_size"), "vocab_size"
        )
        config.input_vocab_size = config.vocab_size
        config.max_seq_len = cls._positive_int(
            cls._required(text, "max_position_embeddings"),
            "max_position_embeddings",
        )
        config.inter_size = cls._positive_int(
            cls._required(text, "intermediate_size"), "intermediate_size"
        )
        config.layernorm_eps = float(text.get("rms_norm_eps", 1e-5))
        if config.layernorm_eps <= 0:
            raise ValueError(
                f"rms_norm_eps must be positive, got {config.layernorm_eps}"
            )
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.has_lm_head = True
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = bool(
            text.get("tie_word_embeddings", outer.get("tie_word_embeddings", False))
        )
        config.config_dtype = text.get("dtype", "bfloat16")
        config.enable_fp32_lm_head = False
        config.has_positional_encoding = False
        config.position_ids_style = 0
        config.qk_norm = False

        eos = text.get("eos_token_id", outer.get("eos_token_id", 0))
        eos_ids = [int(value) for value in eos] if isinstance(eos, list) else [int(eos)]
        if not eos_ids:
            raise ValueError("eos_token_id must contain at least one token id")
        config.special_tokens.eos_token_id = eos_ids[0]
        config.special_tokens.pad_token_id = int(text.get("pad_token_id", eos_ids[0]))
        config.special_tokens.stop_words_id_list = [[value] for value in eos_ids]

    @classmethod
    def _parse_attention(
        cls, text: Dict[str, Any], config: Glm53FlashModelConfig
    ) -> None:
        heads = cls._positive_int(
            cls._required(text, "num_attention_heads"), "num_attention_heads"
        )
        nope_dim = cls._positive_int(
            cls._required(text, "qk_nope_head_dim"), "qk_nope_head_dim"
        )
        rope_dim = int(cls._required(text, "qk_rope_head_dim"))
        if not bool(cls._required(text, "mla_use_nope")) or rope_dim != 0:
            raise ValueError(
                "GLM-5.3-Flash RTP path requires MLA NoPE with qk_rope_head_dim=0"
            )

        attn = config.attn_config
        attn.head_num = heads
        attn.kv_head_num = cls._positive_int(
            text.get("num_key_value_heads", heads), "num_key_value_heads"
        )
        attn.nope_head_dim = nope_dim
        attn.rope_head_dim = rope_dim
        attn.v_head_dim = cls._positive_int(
            cls._required(text, "v_head_dim"), "v_head_dim"
        )
        attn.size_per_head = nope_dim + rope_dim
        attn.q_lora_rank = cls._positive_int(
            cls._required(text, "q_lora_rank"), "q_lora_rank"
        )
        attn.kv_lora_rank = cls._positive_int(
            cls._required(text, "kv_lora_rank"), "kv_lora_rank"
        )
        attn.use_mla = True
        attn.is_causal = True
        attn.rope_config.style = 0
        attn.rope_config.base = 10000
        attn.rope_config.dim = 0
        attn.rope_config.offset = nope_dim
        attn.rope_config.indexer_is_neox_style = not bool(
            text.get("indexer_rope_interleave", True)
        )

        ratio = int(cls._required(text, "index_kpool"))
        raw_topk = int(cls._required(text, "index_topk"))
        if not bool(cls._required(text, "index_kpool_compress")) or ratio != 4:
            raise ValueError(
                "GLM-5.3-Flash requires index_kpool_compress=true and ratio=4"
            )
        if raw_topk <= 0:
            raise ValueError(f"index_topk must be positive, got {raw_topk}")
        if raw_topk % ratio:
            raise ValueError(
                f"index_topk {raw_topk} must be divisible by index_kpool {ratio}"
            )
        attn.is_sparse = True
        attn.indexer_head_dim = cls._positive_int(
            cls._required(text, "index_head_dim"), "index_head_dim"
        )
        attn.indexer_head_num = cls._positive_int(
            cls._required(text, "index_n_heads"), "index_n_heads"
        )
        # Runtime indexer scores pooled history entries. Preserve the checkpoint
        # raw-token budget in Glm53FlashRuntimeConfig below.
        attn.indexer_topk = raw_topk // ratio
        attn.indexer_compress_ratio = ratio
        attn.indexer_compressor_overlap = 0
        attn.sparse_attention_topk = raw_topk + ratio - 1

        linear = cls._required(text, "linear_attn_config")
        if not isinstance(linear, dict):
            raise ValueError("linear_attn_config must be an object")
        linear_heads = cls._positive_int(
            cls._required(linear, "num_heads"), "linear_attn_config.num_heads"
        )
        linear_dim = cls._positive_int(
            cls._required(linear, "head_dim"), "linear_attn_config.head_dim"
        )
        la = config.linear_attention_config
        la.linear_num_key_heads = linear_heads
        la.linear_num_value_heads = linear_heads
        la.linear_key_head_dim = linear_dim
        la.linear_value_head_dim = linear_dim
        la.linear_conv_kernel_dim = cls._positive_int(
            linear.get("short_conv_kernel_size", 4),
            "linear_attn_config.short_conv_kernel_size",
        )

    @classmethod
    def _parse_moe(cls, text: Dict[str, Any], config: Glm53FlashModelConfig) -> None:
        if text.get("hidden_act") != "silu":
            raise ValueError("GLM-5.3-Flash requires hidden_act='silu'")
        if text.get("scoring_func") != "sigmoid":
            raise ValueError("GLM-5.3-Flash requires sigmoid MoE routing")
        config.expert_num = cls._positive_int(
            cls._required(text, "n_routed_experts"), "n_routed_experts"
        )
        config.moe_k = cls._positive_int(
            cls._required(text, "num_experts_per_tok"), "num_experts_per_tok"
        )
        config.moe_inter_size = cls._positive_int(
            cls._required(text, "moe_intermediate_size"), "moe_intermediate_size"
        )
        config.moe_n_group = cls._positive_int(text.get("n_group", 1), "n_group")
        config.moe_topk_group = cls._positive_int(
            text.get("topk_group", 1), "topk_group"
        )
        if config.moe_k > config.expert_num:
            raise ValueError(
                f"num_experts_per_tok {config.moe_k} exceeds "
                f"n_routed_experts {config.expert_num}"
            )
        if config.expert_num % config.moe_n_group:
            raise ValueError(
                f"n_routed_experts {config.expert_num} must be divisible by "
                f"n_group {config.moe_n_group}"
            )
        if config.moe_topk_group > config.moe_n_group:
            raise ValueError(
                f"topk_group {config.moe_topk_group} exceeds n_group "
                f"{config.moe_n_group}"
            )
        config.scoring_func = 1
        config.routed_scaling_factor = float(
            cls._required(text, "routed_scaling_factor")
        )
        if config.routed_scaling_factor <= 0:
            raise ValueError(
                "routed_scaling_factor must be positive, got "
                f"{config.routed_scaling_factor}"
            )
        config.has_moe_norm = bool(text.get("norm_topk_prob", True))
        config.moe_normalize_expert_scale = config.has_moe_norm
        num_shared_experts = cls._positive_int(
            text.get("n_shared_experts", 1), "n_shared_experts"
        )
        config.moe_style = 2
        config.inter_size = num_shared_experts * config.moe_inter_size

        mlp_types = list(cls._required(text, "mlp_layer_types"))
        if len(mlp_types) != config.num_layers:
            raise ValueError("mlp_layer_types must have one entry per decoder layer")
        unknown = sorted(set(mlp_types) - {"dense", "sparse"})
        if unknown:
            raise ValueError(f"unsupported GLM-5.3-Flash MLP layer types: {unknown}")
        config.moe_layer_index = [
            layer
            for layer, layer_type in enumerate(mlp_types)
            if layer_type == "sparse"
        ]

    @classmethod
    def _parse_hybrid(cls, text: Dict[str, Any], config: Glm53FlashModelConfig) -> None:
        linear = cls._required(text, "linear_attn_config")
        kda = cls._zero_based_layer_set(
            cls._required(linear, "kda_layers"), "kda_layers", config.num_layers
        )
        full = cls._zero_based_layer_set(
            cls._required(linear, "full_attn_layers"),
            "full_attn_layers",
            config.num_layers,
        )
        if kda & full:
            raise ValueError(f"KDA/MLA schedules overlap at {sorted(kda & full)}")
        missing = set(range(config.num_layers)) - kda - full
        if missing:
            raise ValueError(f"KDA/MLA schedules miss layers {sorted(missing)}")
        if not kda:
            raise ValueError("GLM-5.3-Flash requires at least one KDA layer")
        if not full:
            raise ValueError("GLM-5.3-Flash requires at least one sparse MLA layer")

        declared = list(cls._required(text, "layer_types"))
        if len(declared) != config.num_layers:
            raise ValueError("layer_types must have one entry per decoder layer")
        for layer, kind in enumerate(declared):
            expected = (
                "linear_attention" if layer in kda else "deepseek_sparse_attention"
            )
            if kind != expected:
                raise ValueError(
                    f"layer_types[{layer}]={kind!r}, expected {expected!r}"
                )

        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.enable_independent_kv_cache_pools = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR if layer in kda else HybridAttentionType.NONE
            for layer in range(config.num_layers)
        ]
        config.attn_config.indexer_layer_ids = sorted(full)

    @classmethod
    def _parse_runtime(
        cls, text: Dict[str, Any], config: Glm53FlashModelConfig
    ) -> None:
        if not bool(cls._required(text, "mhc")):
            raise ValueError("GLM-5.3-Flash RTP path requires mhc=true")
        if not bool(cls._required(text, "index_kpool_always_select_tail")):
            raise ValueError(
                "GLM-5.3-Flash RTP path requires index_kpool_always_select_tail=true"
            )
        linear = cls._required(text, "linear_attn_config")
        ratio = int(cls._required(text, "index_kpool"))
        raw_topk = int(cls._required(text, "index_topk"))
        hc_mult = cls._positive_int(cls._required(text, "hc_mult"), "hc_mult")
        hc_sinkhorn_iters = cls._positive_int(
            cls._required(text, "hc_sinkhorn_iters"), "hc_sinkhorn_iters"
        )
        hc_eps = float(cls._required(text, "hc_eps"))
        if hc_eps <= 0:
            raise ValueError(f"hc_eps must be positive, got {hc_eps}")
        swiglu_limit = float(cls._required(text, "swiglu_limit"))
        if swiglu_limit < 0:
            raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")
        num_shared_experts = cls._positive_int(
            cls._required(text, "n_shared_experts"), "n_shared_experts"
        )
        config.glm5_3_flash_runtime_config = Glm53FlashRuntimeConfig(
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            hc_eps=hc_eps,
            swiglu_limit=swiglu_limit,
            dense_intermediate_size=int(cls._required(text, "intermediate_size")),
            num_shared_experts=num_shared_experts,
            kda_gate_lower_bound=float(cls._required(linear, "gate_lower_bound")),
            # The checkpoint exposes the rank through g_a_proj: [128, hidden].
            kda_gate_lora_rank=128,
            index_raw_topk=raw_topk,
            index_group_topk=raw_topk // ratio,
            index_kpool=ratio,
            index_kpool_always_select_tail=bool(
                cls._required(text, "index_kpool_always_select_tail")
            ),
            index_share_for_mtp_iteration=bool(
                cls._required(text, "index_share_for_mtp_iteration")
            ),
            mla_use_nope=bool(cls._required(text, "mla_use_nope")),
        )
        # Generic MoE consumes this established ModelConfig field. Keeping it
        # alongside the immutable runtime bundle lets the selected executor
        # enforce the released routed-expert activation semantics.
        config.swiglu_limit = config.glm5_3_flash_runtime_config.swiglu_limit

    def support_cuda_graph(self) -> bool:
        # The GLM-5.3-Flash executor is built from the same graph-safe KDA, MLA,
        # HC, and MoE modules used by the K3/DSV4 Python path.  Keep this
        # capability gate in sync with the graph-aware model descriptor,
        # which forwards enable_cuda_graph into every decoder layer.
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.glm5_3_flash import Glm53FlashModel

        self.py_model = Glm53FlashModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        from rtp_llm.models.glm5_3_flash_weight import Glm53FlashWeight

        return Glm53FlashWeight


register_model(
    "glm5_3_flash",
    Glm53Flash,
    ["Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"],
)


__all__ = ["Glm53Flash", "Glm53FlashModelConfig", "Glm53FlashRuntimeConfig"]
