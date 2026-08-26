import json
import os
from typing import Any, Dict

from transformers import AutoTokenizer

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import yarn_get_mscale


class DeepSeekVLV2(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "gated-silu"
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            content = reader.read()
            top_config_json = json.loads(content)
        DeepSeekVLV2._from_hf(config, top_config_json)
        DeepSeekVLV2._load_vit_param(config, top_config_json)
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel

        if self.model_config.attn_config.use_mla:
            raise RuntimeError(
                "DeepSeek-VL2 small/full checkpoints use MLA, which is not "
                "supported by the legacy DeepSeekVLV2Weight layout; do not "
                "force the legacy loader with USE_NEW_LOADER=0"
            )
        self.py_model = GenericMoeModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def _from_hf(config: ModelConfig, top_config_json: Dict[str, Any]):

        config.model_name = "deepseek_vl_v2"
        config.model_type = "deepseek_vl_v2"
        config_json = top_config_json.get("language_config")
        if not isinstance(config_json, dict):
            raise ValueError(
                "DeepSeek-VL2 config.json must contain a language_config object"
            )

        # DeepSeek-VL2 full omits several fields and relies on the official
        # DeepseekV2Config defaults. Keep those defaults explicit here so both
        # legacy and newloader routes build the checkpoint's real topology.
        config.hidden_size = config_json.get("hidden_size", 4096)
        config.num_layers = config_json.get("num_hidden_layers", 30)
        config.attn_config.head_num = config_json.get("num_attention_heads", 32)
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.vocab_size = config_json.get("vocab_size", 102400)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        if config.hidden_size % config.attn_config.head_num:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.attn_config.head_num}"
            )
        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))

        use_mla = config_json.get("use_mla", True)
        if not isinstance(use_mla, bool):
            raise TypeError(f"language_config.use_mla must be bool, got {use_mla!r}")
        config.attn_config.use_mla = use_mla
        if use_mla:
            q_lora_rank = config_json.get("q_lora_rank", 1536)
            config.attn_config.q_lora_rank = (
                0 if q_lora_rank is None else int(q_lora_rank)
            )
            kv_lora_rank = config_json.get("kv_lora_rank", 512)
            config.attn_config.kv_lora_rank = (
                0 if kv_lora_rank is None else int(kv_lora_rank)
            )
            config.attn_config.nope_head_dim = config_json.get("qk_nope_head_dim", 128)
            config.attn_config.rope_head_dim = config_json.get("qk_rope_head_dim", 64)
            config.attn_config.v_head_dim = config_json.get("v_head_dim", 128)
            if (
                config.attn_config.kv_lora_rank <= 0
                or config.attn_config.nope_head_dim <= 0
                or config.attn_config.rope_head_dim <= 0
                or config.attn_config.v_head_dim <= 0
            ):
                raise ValueError("DeepSeek-VL2 MLA dimensions must all be positive")
            config.attn_config.size_per_head = (
                config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
            )
            config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
            config.attn_config.rope_config.offset = config.attn_config.nope_head_dim
            config.attn_config.rope_config.style = (
                5 if config.mla_ops_type == MlaOpsType.MHA else 0
            )
            rope_scaling = config_json.get("rope_scaling")
            if rope_scaling is not None:
                if not isinstance(rope_scaling, dict):
                    raise TypeError("language_config.rope_scaling must be a mapping")
                required_yarn_fields = {
                    "factor",
                    "original_max_position_embeddings",
                    "mscale",
                    "mscale_all_dim",
                }
                missing_yarn_fields = required_yarn_fields - rope_scaling.keys()
                if missing_yarn_fields:
                    raise ValueError(
                        "language_config.rope_scaling is missing required fields: "
                        + ", ".join(sorted(missing_yarn_fields))
                    )
                scaling_factor = float(rope_scaling["factor"])
                mscale = float(rope_scaling["mscale"])
                mscale_all_dim = float(rope_scaling["mscale_all_dim"])
                config.attn_config.rope_config.scale = scaling_factor
                config.attn_config.rope_config.factor1 = float(
                    rope_scaling.get("beta_slow", 1)
                )
                config.attn_config.rope_config.factor2 = float(
                    rope_scaling.get("beta_fast", 32)
                )
                config.attn_config.rope_config.max_pos = int(
                    rope_scaling["original_max_position_embeddings"]
                )
                config.deepseek_rope_mscale = mscale
                config.deepseek_mscale_all_dim = mscale_all_dim
                config.attn_config.rope_config.mscale = yarn_get_mscale(
                    scaling_factor, mscale
                ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

            rope_interleave = config_json.get("rope_interleave", True)
            if not isinstance(rope_interleave, bool):
                raise TypeError("language_config.rope_interleave must be bool")
            config.attn_config.rope_config.is_neox_style = not rope_interleave
            indexer_rope_interleave = config_json.get("indexer_rope_interleave", False)
            if not isinstance(indexer_rope_interleave, bool):
                raise TypeError("language_config.indexer_rope_interleave must be bool")
            config.attn_config.rope_config.indexer_is_neox_style = (
                not indexer_rope_interleave
            )
        else:
            config.attn_config.size_per_head = (
                config.hidden_size // config.attn_config.head_num
            )
            config.attn_config.rope_config.dim = config.attn_config.size_per_head
            config.attn_config.rope_config.style = 1

        # from Llama
        config.layernorm_eps = config_json.get(
            "rms_norm_eps", config_json.get("layer_norm_eps", 1e-6)
        )

        # MOE config
        if "scoring_func" in config_json:
            scoring_func = config_json["scoring_func"]
            if scoring_func == "softmax":
                config.scoring_func = 0
            elif scoring_func == "sigmoid":
                config.scoring_func = 1
            else:
                raise ValueError(f"Unknown scoring_func: {scoring_func}")
        else:
            # default is softmax
            config.scoring_func = 0

        config.routed_scaling_factor = config_json.get("routed_scaling_factor", 1.0)
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["n_routed_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.moe_n_group = config_json.get("n_group", 1)
        config.moe_topk_group = config_json.get("topk_group", 1)

        n_shared_experts = config_json["n_shared_experts"]
        config.inter_size = n_shared_experts * config.moe_inter_size

        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + expert

        moe_step = config_json.get("moe_layer_freq", 1)
        first_k_dense_replace = config_json["first_k_dense_replace"]
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

        config.config_dtype = config_json.get("torch_dtype", None)

        if config.special_tokens is None:
            from rtp_llm.config.model_config import SpecialTokens

            config.special_tokens = SpecialTokens()
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 1)
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", 0)

    @staticmethod
    def _load_vit_param(config: ModelConfig, top_config_json: Dict[str, Any]):
        vision_config = top_config_json.get("vision_config", {})
        config.mm_related_params.config["vision_config"] = vision_config
        projector_config = top_config_json.get("projector_config", {})
        config.mm_related_params.config["projector_config"] = projector_config
        candidate_resolutions = top_config_json.get(
            "candidate_resolutions", ((384, 384),)
        )
        config.mm_related_params.config["candidate_resolutions"] = candidate_resolutions
        config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})
        config.mm_related_params.config["tile_tag"] = top_config_json.get(
            "tile_tag", "2D"
        )
        config.mm_related_params.config["global_view_pos"] = top_config_json.get(
            "global_view_pos", "head"
        )

        tokenizer = AutoTokenizer.from_pretrained(config.ckpt_path)
        image_id = tokenizer.encode("<image>", add_special_tokens=False)[0]
        config.mm_related_params.special_token_ids.update(
            {"ignore_token_index": -100, "image_token_index": image_id}
        )
        config.mm_model_config.mm_sep_tokens = [[image_id]]
        config.mm_model_config.is_multimodal = True

    @staticmethod
    def get_weight_cls():
        # Keep the legacy weight graph out of the newloader import path; only
        # the old loader asks for this class.
        from rtp_llm.models.deepseek_vl2.deepseek_vl2_weight import DeepSeekVLV2Weight

        return DeepSeekVLV2Weight


register_model("deepseek_vl_v2", DeepSeekVLV2, ["DeepseekVL2ForCausalLM"])
