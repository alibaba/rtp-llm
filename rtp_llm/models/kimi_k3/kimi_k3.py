import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3Eagle3Weight, KimiK3Weight
from rtp_llm.ops import HybridAttentionType


@dataclass(frozen=True)
class KimiK3RuntimeConfig:
    """Checkpoint fields consumed only by the Python Kimi K3 implementation."""

    dense_intermediate_size: int
    routed_expert_hidden_size: int
    num_shared_experts: int
    activation_situ_beta: float
    activation_situ_linear_beta: Optional[float]
    attn_res_block_size: int
    latent_moe_use_norm: bool
    mla_use_nope: bool
    mla_use_output_gate: bool
    kda_gate_lower_bound: Optional[float]
    kda_use_full_rank_gate: bool


class KimiK3ModelConfig(ModelConfig):
    """RTP model config with a K3-scoped, strongly typed Python extension."""

    _python_fields = ModelConfig._python_fields | {"k3_runtime_config"}
    k3_runtime_config: KimiK3RuntimeConfig

    def disables_framework_deepep_moe(self) -> bool:
        """K3 MegaMoE performs dispatch/combine through symmetric memory."""
        return True

    def init_precision_config(
        self, kv_cache_config: Optional[Any], act_type: Optional[str]
    ) -> None:
        super().init_precision_config(kv_cache_config, act_type)
        if self.compute_dtype != torch.bfloat16:
            raise ValueError(
                "Kimi K3 currently supports only BF16 compute, got "
                f"{self.compute_dtype}"
            )
        if self.quant_config is not None:
            raise ValueError(
                "Kimi K3 does not support runtime weight quantization; its "
                "checkpoint-native MXFP4 experts are loaded by the K3 weight path"
            )


class KimiK3(BaseModel):
    """Kimi K3 text model with MoonViT multimodal support.

    Kimi K3 checkpoints use a multimodal outer config and put the complete text
    model config under ``text_config``.  The MoonViT vision tower and projector
    live in the registry-based mixin ``KimiK3Mixin`` (registered for
    ``model_type == "kimi_k3"``); this class only parses config.
    """

    WEIGHT_PREFIX = "language_model."

    @staticmethod
    def _load_vision_config(config: ModelConfig, top_config: Dict[str, Any]) -> None:
        if "vision_config" not in top_config:
            config.mm_model_config.is_multimodal = False
            return

        vision_config = dict(top_config["vision_config"])
        vision_config.pop("_name_or_path", None)
        if "media_placeholder_token_id" not in top_config:
            raise ValueError(
                "Kimi K3 config has vision_config but no media_placeholder_token_id; "
                "guessing it would silently mis-place every image placeholder"
            )
        media_token_id = int(top_config["media_placeholder_token_id"])

        config.mm_model_config.is_multimodal = True
        config.mm_model_config.mm_sep_tokens = [[media_token_id]]
        config.mm_related_params.config = {"vision_config": vision_config}
        config.mm_related_params.special_token_ids.update(
            {"image_token_index": media_token_id}
        )
        config.mm_related_params.special_tokens.update(
            {"default_mm_token": "<|media_pad|>"}
        )
        config.mm_related_params.support_batch = True

    @classmethod
    def _create_config(cls, ckpt_path: str) -> KimiK3ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path, encoding="utf-8") as reader:
            config_json = json.load(reader)
        return cls._from_config_json(config_json, ckpt_path)

    @classmethod
    def _from_config_json(
        cls, config_json: Dict[str, Any], ckpt_path: str = ""
    ) -> KimiK3ModelConfig:
        text_config = config_json.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Kimi K3 config must contain an object-valued text_config")

        model_type = config_json.get("model_type")
        if model_type != "kimi_k3":
            raise ValueError(f"expected outer model_type='kimi_k3', got {model_type!r}")
        if text_config.get("model_type") != "kimi_linear":
            raise ValueError(
                "expected text_config.model_type='kimi_linear', got "
                f"{text_config.get('model_type')!r}"
            )

        config = KimiK3ModelConfig()
        config.ckpt_path = ckpt_path
        config.tokenizer_path = ckpt_path
        config.model_type = "kimi_k3"

        cls._parse_basic_config(config_json, text_config, config)
        cls._parse_attention_config(text_config, config)
        cls._parse_moe_config(text_config, config)
        cls._parse_hybrid_attention_config(text_config, config)
        cls._parse_kimi_runtime_config(text_config, config)
        cls._load_vision_config(config, config_json)

        logging.info(
            "Kimi K3 text config loaded: layers=%d hidden=%d heads=%d "
            "kda_layers=%d mla_layers=%d experts=%d topk=%d attn_res_block=%d",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            sum(
                layer_type == HybridAttentionType.LINEAR
                for layer_type in config.hybrid_attention_config.hybrid_attention_types
            ),
            sum(
                layer_type == HybridAttentionType.NONE
                for layer_type in config.hybrid_attention_config.hybrid_attention_types
            ),
            config.expert_num,
            config.moe_k,
            config.k3_runtime_config.attn_res_block_size,
        )
        return config

    @staticmethod
    def _required(config: Dict[str, Any], name: str) -> Any:
        if name not in config:
            raise ValueError(f"Kimi K3 text_config is missing required field {name!r}")
        return config[name]

    @classmethod
    def _parse_basic_config(
        cls,
        outer_config: Dict[str, Any],
        text_config: Dict[str, Any],
        config: KimiK3ModelConfig,
    ) -> None:
        config.num_layers = int(cls._required(text_config, "num_hidden_layers"))
        config.hidden_size = int(cls._required(text_config, "hidden_size"))
        config.vocab_size = int(cls._required(text_config, "vocab_size"))
        config.input_vocab_size = config.vocab_size
        config.max_seq_len = int(cls._required(text_config, "max_position_embeddings"))
        config.inter_size = int(cls._required(text_config, "intermediate_size"))
        config.layernorm_eps = float(text_config.get("rms_norm_eps", 1e-5))
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.has_lm_head = True
        # C++ only needs to know that the FFN is gated.  The Python K3 modules
        # implement SiTU instead of dispatching the framework's SiGLU kernel.
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = bool(
            text_config.get(
                "tie_word_embeddings", outer_config.get("tie_word_embeddings", False)
            )
        )
        config.config_dtype = text_config.get(
            "dtype", outer_config.get("dtype", "bfloat16")
        )
        config.enable_fp32_lm_head = False
        config.has_positional_encoding = False
        config.position_ids_style = 0
        config.qk_norm = False

        config.special_tokens.bos_token_id = int(
            outer_config.get("bos_token_id", text_config.get("bos_token_id", -1))
        )
        config.special_tokens.eos_token_id = int(
            outer_config.get("eos_token_id", text_config.get("eos_token_id", 0))
        )
        config.special_tokens.pad_token_id = int(
            outer_config.get("pad_token_id", text_config.get("pad_token_id", 0))
        )
        config.special_tokens.stop_words_id_list = [
            [config.special_tokens.eos_token_id]
        ]

    @classmethod
    def _parse_attention_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        head_num = int(cls._required(text_config, "num_attention_heads"))
        qk_nope_head_dim = int(cls._required(text_config, "qk_nope_head_dim"))
        qk_rope_head_dim = int(cls._required(text_config, "qk_rope_head_dim"))

        config.attn_config.head_num = head_num
        config.attn_config.kv_head_num = int(
            text_config.get("num_key_value_heads", head_num)
        )
        config.attn_config.size_per_head = qk_nope_head_dim + qk_rope_head_dim
        config.attn_config.q_lora_rank = int(text_config.get("q_lora_rank") or 0)
        config.attn_config.kv_lora_rank = int(
            cls._required(text_config, "kv_lora_rank")
        )
        config.attn_config.nope_head_dim = qk_nope_head_dim
        # This is a physical 64-d suffix in K3.  The runtime config below marks
        # it as no-RoPE even though existing MLA APIs retain the historical name.
        config.attn_config.rope_head_dim = qk_rope_head_dim
        config.attn_config.v_head_dim = int(cls._required(text_config, "v_head_dim"))
        config.attn_config.use_mla = True
        config.attn_config.is_causal = True
        config.attn_config.rope_config.style = 0
        config.attn_config.rope_config.base = 10000
        config.attn_config.rope_config.dim = qk_rope_head_dim
        config.attn_config.rope_config.offset = qk_nope_head_dim

        linear_config = cls._required(text_config, "linear_attn_config")
        if not isinstance(linear_config, dict):
            raise ValueError("linear_attn_config must be an object")
        linear_heads = int(cls._required(linear_config, "num_heads"))
        if linear_heads != head_num:
            raise ValueError(
                "KDA and MLA head counts must match for hybrid TP: "
                f"KDA={linear_heads}, MLA={head_num}"
            )
        linear_head_dim = int(cls._required(linear_config, "head_dim"))
        config.linear_attention_config.linear_key_head_dim = linear_head_dim
        config.linear_attention_config.linear_value_head_dim = linear_head_dim
        config.linear_attention_config.linear_num_key_heads = linear_heads
        config.linear_attention_config.linear_num_value_heads = linear_heads
        config.linear_attention_config.linear_conv_kernel_dim = int(
            linear_config.get("short_conv_kernel_size", 4)
        )

    @classmethod
    def _parse_moe_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        if text_config.get("hidden_act") != "situ":
            raise ValueError(
                "Kimi K3 requires hidden_act='situ', got "
                f"{text_config.get('hidden_act')!r}"
            )
        if text_config.get("moe_router_activation_func", "sigmoid") != "sigmoid":
            raise ValueError("Kimi K3 bring-up supports only sigmoid MoE routing")

        config.expert_num = int(cls._required(text_config, "num_experts"))
        config.moe_k = int(cls._required(text_config, "num_experts_per_token"))
        if config.moe_k <= 0 or config.moe_k > config.expert_num:
            raise ValueError(
                f"invalid Kimi K3 top-k {config.moe_k} for {config.expert_num} experts"
            )
        config.moe_inter_size = int(cls._required(text_config, "moe_intermediate_size"))
        config.moe_n_group = int(text_config.get("num_expert_group", 1))
        config.moe_topk_group = int(text_config.get("topk_group", 1))
        config.scoring_func = 1  # sigmoid
        config.routed_scaling_factor = float(
            text_config.get("routed_scaling_factor", 1.0)
        )
        config.has_moe_norm = bool(text_config.get("moe_renormalize", True))
        config.moe_normalize_expert_scale = config.has_moe_norm
        config.moe_style = 2  # shared + routed experts

        num_shared_experts = int(text_config.get("num_shared_experts", 0))
        config.inter_size = max(1, num_shared_experts) * config.moe_inter_size
        first_dense = int(text_config.get("first_k_dense_replace", 0))
        moe_frequency = int(text_config.get("moe_layer_freq", 1))
        if first_dense < 0 or moe_frequency <= 0:
            raise ValueError(
                "first_k_dense_replace must be >= 0 and moe_layer_freq must be > 0"
            )
        config.moe_layer_index = [
            layer_idx
            for layer_idx in range(config.num_layers)
            if layer_idx >= first_dense and layer_idx % moe_frequency == 0
        ]

    @classmethod
    def _parse_hybrid_attention_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        linear_config = cls._required(text_config, "linear_attn_config")
        kda_layers = cls._one_based_layer_set(
            linear_config.get("kda_layers", []), "kda_layers", config.num_layers
        )
        mla_layers = cls._one_based_layer_set(
            linear_config.get("full_attn_layers", []),
            "full_attn_layers",
            config.num_layers,
        )
        overlap = kda_layers & mla_layers
        if overlap:
            raise ValueError(
                f"KDA and MLA layer schedules overlap at 1-based layers {sorted(overlap)}"
            )
        expected = set(range(1, config.num_layers + 1))
        missing = expected - kda_layers - mla_layers
        if missing:
            raise ValueError(
                "KDA/MLA schedules must cover every text layer; missing 1-based "
                f"layers {sorted(missing)}"
            )

        config.hybrid_attention_config.enable_hybrid_attention = True
        # MLA and KDA have different cache shapes and lifetimes.  Keep them in
        # independent physical pools; speculative models append their own
        # third pool in CacheConfigCreator instead of sharing either target
        # pool.
        config.hybrid_attention_config.enable_independent_kv_cache_pools = True
        layer_types: List[HybridAttentionType] = []
        for layer_1based in range(1, config.num_layers + 1):
            layer_types.append(
                HybridAttentionType.LINEAR
                if layer_1based in kda_layers
                else HybridAttentionType.NONE
            )
        config.hybrid_attention_config.hybrid_attention_types = layer_types

    @staticmethod
    def _one_based_layer_set(
        values: Iterable[Any], name: str, num_layers: int
    ) -> Set[int]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must contain integer 1-based layer ids")
        try:
            raw_layers = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer 1-based layer ids") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_layers
        ):
            raise ValueError(f"{name} must contain integer 1-based layer ids")
        layers = set(raw_layers)
        invalid = sorted(layer for layer in layers if layer < 1 or layer > num_layers)
        if invalid:
            raise ValueError(
                f"{name} contains out-of-range 1-based layers {invalid}; "
                f"valid range is [1, {num_layers}]"
            )
        return layers

    @classmethod
    def _parse_kimi_runtime_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        linear_config = cls._required(text_config, "linear_attn_config")
        attn_res_block_size = int(cls._required(text_config, "attn_res_block_size"))
        if attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be > 0")

        # SiTU betas are model-defining constants (K3 uses 4.0 / 25.0); a wrong
        # default silently corrupts the activation, so require them explicitly.
        # ``activation_situ_linear_beta`` may still be an explicit null, which
        # legitimately disables the up-projection tanh clamp.
        linear_beta = cls._required(text_config, "activation_situ_linear_beta")
        gate_lower_bound = linear_config.get("gate_lower_bound")
        config.k3_runtime_config = KimiK3RuntimeConfig(
            dense_intermediate_size=int(
                cls._required(text_config, "intermediate_size")
            ),
            routed_expert_hidden_size=int(
                cls._required(text_config, "routed_expert_hidden_size")
            ),
            num_shared_experts=int(text_config.get("num_shared_experts", 0)),
            activation_situ_beta=float(
                cls._required(text_config, "activation_situ_beta")
            ),
            activation_situ_linear_beta=(
                None if linear_beta is None else float(linear_beta)
            ),
            attn_res_block_size=attn_res_block_size,
            latent_moe_use_norm=bool(text_config.get("latent_moe_use_norm", False)),
            mla_use_nope=bool(cls._required(text_config, "mla_use_nope")),
            mla_use_output_gate=bool(cls._required(text_config, "mla_use_output_gate")),
            kda_gate_lower_bound=(
                None if gate_lower_bound is None else float(gate_lower_bound)
            ),
            kda_use_full_rank_gate=bool(linear_config.get("use_full_rank_gate", False)),
        )

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model

        # Release inactive blocks left by checkpoint loading before the expert
        # weight transform allocates its large temporary buffer.
        torch.cuda.empty_cache()

        self.py_model = KimiK3Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        # Release rank-dependent inactive blocks before the native engine
        # allocates KV cache and workspaces.
        torch.cuda.empty_cache()
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return KimiK3Weight


class KimiK3Eagle3(KimiK3):
    """One-layer MLA/SWA EAGLE-3 draft model for Kimi K3."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> KimiK3ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path, encoding="utf-8") as reader:
            raw = json.load(reader)

        if raw.get("model_type") != "deepseek_v3_swa":
            raise ValueError(
                "Kimi K3 EAGLE-3 expects model_type='deepseek_v3_swa', got "
                f"{raw.get('model_type')!r}"
            )
        if int(raw.get("num_hidden_layers", 0)) != 1:
            raise ValueError("Kimi K3 EAGLE-3 currently requires exactly one layer")

        config = KimiK3ModelConfig()
        config.ckpt_path = ckpt_path
        config.tokenizer_path = ckpt_path
        config.model_type = "kimi_k3_mla_swa_eagle3"
        config.num_layers = 1
        config.hidden_size = int(cls._required(raw, "hidden_size"))
        config.vocab_size = int(cls._required(raw, "vocab_size"))
        config.input_vocab_size = config.vocab_size
        config.max_seq_len = int(cls._required(raw, "max_position_embeddings"))
        config.inter_size = int(cls._required(raw, "intermediate_size"))
        config.layernorm_eps = float(raw.get("rms_norm_eps", 1e-5))
        config.norm_type = "rmsnorm"
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.has_lm_head = True
        config.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
        config.config_dtype = raw.get("torch_dtype", "bfloat16")
        config.has_positional_encoding = False
        config.position_ids_style = 0
        config.qk_norm = False
        config.moe_layer_index = []
        config.expert_num = 0
        config.moe_k = 0

        head_num = int(cls._required(raw, "num_attention_heads"))
        config.attn_config.head_num = head_num
        config.attn_config.kv_head_num = int(raw.get("num_key_value_heads", head_num))
        config.attn_config.nope_head_dim = int(cls._required(raw, "qk_nope_head_dim"))
        config.attn_config.rope_head_dim = int(cls._required(raw, "qk_rope_head_dim"))
        config.attn_config.size_per_head = (
            config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
        )
        config.attn_config.v_head_dim = int(cls._required(raw, "v_head_dim"))
        config.attn_config.q_lora_rank = int(cls._required(raw, "q_lora_rank"))
        config.attn_config.kv_lora_rank = int(cls._required(raw, "kv_lora_rank"))
        config.attn_config.sliding_window = int(cls._required(raw, "sliding_window"))
        config.attn_config.use_mla = True
        config.attn_config.is_causal = True
        config.attn_config.rope_config.style = 0
        config.attn_config.rope_config.base = int(raw.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim
        config.attn_config.rope_config.is_neox_style = False

        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.enable_independent_kv_cache_pools = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.SLIDING_WINDOW
        ]
        config.k3_runtime_config = KimiK3RuntimeConfig(
            dense_intermediate_size=config.inter_size,
            routed_expert_hidden_size=0,
            num_shared_experts=0,
            activation_situ_beta=0.0,
            activation_situ_linear_beta=None,
            attn_res_block_size=1,
            latent_moe_use_norm=False,
            mla_use_nope=bool(raw.get("mla_use_nope", False)),
            mla_use_output_gate=bool(raw.get("mla_use_output_gate", True)),
            kda_gate_lower_bound=None,
            kda_use_full_rank_gate=False,
        )
        config.special_tokens.bos_token_id = int(raw.get("bos_token_id", -1))
        config.special_tokens.eos_token_id = int(raw.get("eos_token_id", 0))
        config.special_tokens.pad_token_id = int(raw.get("pad_token_id", 0))
        config.special_tokens.stop_words_id_list = [
            [config.special_tokens.eos_token_id]
        ]
        return config

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.kimi_k3_eagle3 import KimiK3Eagle3Model

        self.py_model = KimiK3Eagle3Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return KimiK3Eagle3Weight


register_model(
    "kimi_k3",
    KimiK3,
    ["KimiK3ForConditionalGeneration"],
)
register_model(
    "kimi_k3_mla_swa_eagle3",
    KimiK3Eagle3,
    ["Eagle3DeepseekV2SWAForCausalLM"],
)
