import json
import math
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.gemma4_weight import Gemma4WeightInfo
from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheCpPolicyDesc,
    CacheGroupType,
    CacheTailPolicyDesc,
    CpBlockMappingMode,
    CpBlockSliceMode,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
)

# KV cache group tags shared with the python model descriptor (frozen interface).
GEMMA4_FULL_ATTENTION_TAG = "full"
GEMMA4_SWA_TAG = "swa"

# The merged VLM config nests the text model under "text_config"; the top-level
# object only adds vision/audio fields which the text-only path ignores.
GEMMA4_TEXT_CONFIG_KEY = "text_config"


class Gemma4(BaseModel):
    """Gemma4 text model (first step: pure-text path, vision weights ignored).

    Layer facts (verified against gemma-4-26B-A4B-it):
    - 25 sliding-attention layers (SWA window 1024, 16Q x 8KV x 256,
      RoPE theta 1e4 on all 256 dims) + 5 full-attention layers
      (5/11/17/23/29, 16Q x 2KV x 512, K==V with no v_proj).
    - Global attn_config carries the sliding geometry; full layers override
      kv_head_num/size_per_head per KVCacheSpecDesc once the pybind fields land.
    - Every layer runs a dense MLP (inter_size 2112) in parallel with routed
      experts (128 experts, top-8, moe_inter_size 704).
    - All RMSNorms multiply the weight directly (no +1 shift, unlike Gemma1-3).
    """

    @staticmethod
    def get_weight_cls():
        return Gemma4WeightInfo

    @classmethod
    def _load_config_json(cls, ckpt_path: str) -> Dict[str, Any]:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            return json.loads(reader.read())

    @classmethod
    def _load_text_config(cls, ckpt_path: str) -> Dict[str, Any]:
        config_json = cls._load_config_json(ckpt_path)
        text_config = config_json.get(GEMMA4_TEXT_CONFIG_KEY)
        if isinstance(text_config, dict):
            return text_config
        return config_json

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_json = cls._load_config_json(ckpt_path)
        text_config = config_json.get(GEMMA4_TEXT_CONFIG_KEY)
        if not isinstance(text_config, dict):
            text_config = config_json

        cls._validate_text_config(text_config)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        # --- geometry: global values describe the sliding layers ---
        config.num_layers = text_config["num_hidden_layers"]
        config.hidden_size = text_config["hidden_size"]
        config.vocab_size = text_config["vocab_size"]
        config.attn_config.head_num = text_config["num_attention_heads"]
        config.attn_config.kv_head_num = text_config.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = text_config["head_dim"]
        config.attn_config.sliding_window = int(text_config.get("sliding_window", 0))
        # default page size; build_model_config overrides with the runtime
        # kv_cache_config.seq_size_per_block when provided
        config.attn_config.tokens_per_block = 128

        # --- rope: global config carries the sliding-layer rope only ---
        # Full layers apply their own partial RoPE (theta 1e6, first 25% dims)
        # inside the python model descriptor, independent of this global entry.
        rope_parameters = text_config.get("rope_parameters") or {}
        sliding_rope = rope_parameters.get("sliding_attention") or {}
        config.attn_config.rope_config.style = 1  # RopeStyle::Base
        config.attn_config.rope_config.base = int(sliding_rope.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = config.attn_config.size_per_head

        # --- normalization / activation ---
        config.norm_type = "rmsnorm"
        config.layernorm_eps = text_config.get("rms_norm_eps", 1e-6)
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = True
        # hidden_activation == gelu_pytorch_tanh -> gated GELU (tanh approx)
        config.activation_type = "geglu"

        # --- MoE: dense MLP + routed experts on every layer ---
        # moe_style=2 with moe_layer_index covering all layers makes
        # ModelConfig.layer_weight_param_count() account for both the dense MLP
        # (inter_size) and the routed experts (moe_inter_size) of each layer.
        config.inter_size = text_config["intermediate_size"]
        config.moe_inter_size = text_config["moe_intermediate_size"]
        config.moe_w1_layout = "gate_up"
        config.expert_num = text_config["num_experts"]
        config.moe_k = text_config.get(
            "top_k_experts", text_config.get("num_experts_per_tok", 0)
        )
        # The parallel dense MLP occupies the shared-expert slot of the config;
        # the python descriptor executes the real sandwich structure.
        config.n_shared_experts = 1 if config.inter_size > 0 else 0
        config.moe_style = 2 if config.n_shared_experts > 0 else 1
        # router renormalizes the top-k probabilities (norm_topk_prob)
        config.has_moe_norm = True
        config.moe_layer_index = list(range(config.num_layers))

        # --- hybrid attention layer types ---
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = cls._parse_layer_types(
            text_config, config.num_layers
        )
        # Full-attention geometry and RoPE stay on this ModelConfig instance so
        # independent checkpoints cannot overwrite each other's layer contract.
        full_rope = rope_parameters.get("full_attention") or {}
        config.mm_related_params.config.update(
            {
                "full_layer_kv_head_num": int(
                    text_config.get("num_global_key_value_heads")
                    or config.attn_config.kv_head_num
                ),
                "full_layer_size_per_head": int(
                    text_config.get("global_head_dim", 512)
                ),
                "full_layer_rope_theta": float(
                    full_rope.get("rope_theta", 1_000_000.0)
                ),
                "full_layer_partial_rotary_factor": float(
                    full_rope.get("partial_rotary_factor", 0.25)
                ),
            }
        )

        # --- misc ---
        config.tie_word_embeddings = text_config.get("tie_word_embeddings", False)
        config.enable_fp32_lm_head = False
        config.max_seq_len = int(text_config.get("max_position_embeddings", 8192))

        config.special_tokens.bos_token_id = int(text_config.get("bos_token_id", 2))
        eos_token_ids = config_json.get(
            "eos_token_id", text_config.get("eos_token_id", 1)
        )
        generation_path = os.path.join(ckpt_path, "generation_config.json")
        if os.path.exists(generation_path):
            with open(generation_path) as reader:
                generation_config = json.loads(reader.read())
            eos_token_ids = generation_config.get("eos_token_id", eos_token_ids)
        if not isinstance(eos_token_ids, (list, tuple)):
            eos_token_ids = [eos_token_ids]
        eos_token_ids = [int(token_id) for token_id in eos_token_ids]
        config.special_tokens.eos_token_id = eos_token_ids[0]
        config.special_tokens.stop_words_id_list = [
            [token_id] for token_id in eos_token_ids
        ]
        pad_token_id = text_config.get("pad_token_id")
        if pad_token_id is not None:
            config.special_tokens.pad_token_id = int(pad_token_id)

        vision_config = config_json.get("vision_config")
        if isinstance(vision_config, dict):
            config.mm_related_params.config.update(vision_config)
            config.mm_related_params.config["ckpt_path"] = ckpt_path
            if config.mm_related_params.special_tokens is None:
                from rtp_llm.config.model_config import SpecialTokens

                config.mm_related_params.special_tokens = SpecialTokens()
            config.mm_related_params.special_tokens.update(
                {"default_mm_token": "<|image|>"}
            )
            boi_token_id = int(config_json.get("boi_token_id", 255999))
            eoi_token_id = int(config_json.get("eoi_token_id", 258882))
            config.mm_model_config.mm_sep_tokens = [[boi_token_id, eoi_token_id]]
            config.mm_model_config.is_multimodal = True

        config.config_dtype = text_config.get("torch_dtype") or text_config.get("dtype")

        config.final_logit_softcapping = float(
            text_config.get("final_logit_softcapping", 0.0)
        )

        return config

    @staticmethod
    def _validate_text_config(text: Dict[str, Any]) -> None:
        positive = (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts",
        )
        for name in positive:
            value = text.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Gemma4 {name} must be a positive integer")
        q_heads = text["num_attention_heads"]
        for name, default in (
            ("num_key_value_heads", q_heads),
            ("num_global_key_value_heads", text["num_key_value_heads"]),
        ):
            heads = text.get(name, default)
            if type(heads) is not int or heads <= 0 or q_heads % heads:
                raise ValueError(f"Gemma4 {name} must divide num_attention_heads")
        for name, default in (("head_dim", 256), ("global_head_dim", 512)):
            dim = text.get(name, default)
            if type(dim) is not int or dim <= 0 or dim % 2:
                raise ValueError(f"Gemma4 {name} must be a positive even integer")
        k = text.get("top_k_experts", text.get("num_experts_per_tok", 0))
        if type(k) is not int or not 0 < k <= text["num_experts"]:
            raise ValueError("Gemma4 top_k_experts must be in [1, num_experts]")
        eps = float(text.get("rms_norm_eps", 1e-6))
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Gemma4 rms_norm_eps must be finite and positive")
        cap = float(text.get("final_logit_softcapping", 0.0))
        if not math.isfinite(cap) or cap < 0:
            raise ValueError(
                "Gemma4 final_logit_softcapping must be finite and non-negative"
            )
        types = text.get("layer_types", [])
        if any(v in ("sliding_attention", "sliding") for v in types):
            window = text.get("sliding_window", 0)
            if type(window) is not int or window <= 0:
                raise ValueError("Gemma4 sliding_window must be a positive integer")
        rope = text.get("rope_parameters") or {}
        for name, expected_type in (
            ("sliding_attention", "default"),
            ("full_attention", "proportional"),
        ):
            entry = rope.get(name) or {}
            if entry.get("rope_type", expected_type) != expected_type:
                raise ValueError(f"unsupported Gemma4 {name} rope_type")
            base = float(
                entry.get(
                    "rope_theta", 10000 if name == "sliding_attention" else 1000000
                )
            )
            if not math.isfinite(base) or base <= 0:
                raise ValueError("Gemma4 rope_theta must be finite and positive")
        factor = float(
            (rope.get("full_attention") or {}).get("partial_rotary_factor", 0.25)
        )
        if not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("Gemma4 full partial_rotary_factor must be in (0, 1]")
        if text.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Gemma4 requires gelu_pytorch_tanh")
        if not text.get("attention_k_eq_v", True):
            raise ValueError("Gemma4 FULL attention requires attention_k_eq_v")
        if text.get("hidden_size_per_layer_input", 0) or text.get(
            "num_kv_shared_layers", 0
        ):
            raise ValueError(
                "Gemma4 per-layer inputs and target KV sharing are not supported"
            )

    @staticmethod
    def _parse_layer_types(
        text_config: Dict[str, Any], num_layers: int
    ) -> List[HybridAttentionType]:
        layer_types = text_config.get("layer_types")
        if not layer_types:
            # Without a per-layer declaration every layer attends globally.
            return [HybridAttentionType.NONE] * num_layers
        if len(layer_types) != num_layers:
            raise ValueError(
                f"gemma4 layer_types has {len(layer_types)} entries but "
                f"num_hidden_layers is {num_layers}"
            )
        hybrid_types: List[HybridAttentionType] = []
        for entry in layer_types:
            if entry in ("full_attention", "full"):
                hybrid_types.append(HybridAttentionType.NONE)
            elif entry in ("sliding_attention", "sliding"):
                hybrid_types.append(HybridAttentionType.SLIDING_WINDOW)
            else:
                raise ValueError(f"unsupported gemma4 layer_types entry: {entry}")
        return hybrid_types

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return

        hybrid_types = list(model_config.hybrid_attention_config.hybrid_attention_types)
        if not hybrid_types:
            hybrid_types = [
                HybridAttentionType.SLIDING_WINDOW
            ] * model_config.num_layers
        if len(hybrid_types) != model_config.num_layers:
            raise ValueError(
                f"gemma4 hybrid_attention_types has {len(hybrid_types)} entries "
                f"but num_layers is {model_config.num_layers}"
            )

        gemma4_config = model_config.mm_related_params.config
        full_kv_head_num = int(gemma4_config["full_layer_kv_head_num"])
        full_size_per_head = int(gemma4_config["full_layer_size_per_head"])

        layer_descs: List[List[KVCacheSpecDesc]] = []
        for attn_type in hybrid_types:
            desc = KVCacheSpecDesc()
            desc.cache_type = KVCacheSpecType.MHA
            cp = CacheCpPolicyDesc()
            cp.mapping = CpBlockMappingMode.BLOCK_ROUND_ROBIN
            cp.slice = CpBlockSliceMode.NONE
            desc.cp = cp
            if attn_type == HybridAttentionType.NONE:
                desc.tag = GEMMA4_FULL_ATTENTION_TAG
                desc.group_type = CacheGroupType.FULL
                desc.kv_head_num = full_kv_head_num
                desc.size_per_head = full_size_per_head
            else:
                desc.tag = GEMMA4_SWA_TAG
                desc.group_type = CacheGroupType.SWA
                desc.kv_head_num = model_config.attn_config.kv_head_num
                desc.size_per_head = model_config.attn_config.size_per_head
                window = int(model_config.attn_config.sliding_window)
                block = int(model_config.attn_config.tokens_per_block)
                if block <= 0:
                    raise ValueError("Gemma4 SWA requires a positive cache block size")
                tail_desc = CacheTailPolicyDesc()
                if window > 0:
                    tail_desc.active_tail_blocks = (window + block - 1) // block + 1
                    capacity_desc = CacheCapacityPolicyDesc()
                    capacity_desc.reservable = False
                    capacity_desc.bounded_by_active_tail = True
                    desc.capacity = capacity_desc
                desc.tail = tail_desc
            layer_descs.append([desc])
        model_config.kv_cache_spec_descs = layer_descs

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.gemma4 import Gemma4Model

        self.py_model = Gemma4Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            moe_config=self.moe_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True


register_model(
    "gemma4", Gemma4, ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"]
)
