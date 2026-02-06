import json
import os
from typing import Any, Dict, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_vl2.deepseek_vl2_vit import DeepSeekVLV2ImageEmbedding
from rtp_llm.models.deepseek_vl2.deepseek_vl2_weight import (
    DeepSeekVLV2VitWeight,
    DeepSeekVLV2Weight,
)
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase


class DeepSeekVLV2(BaseModel, MultiModalMixin):

    def _init_multimodal(self, mm_model_config, vit_config):
        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = self.model_config.mm_related_params
        self.ignore_id = -100
        self.mm_part = DeepSeekVLV2ImageEmbedding(
            mm_related_params,
            model_config=self.model_config,
            ignore_id=self.ignore_id,
        )
        mm_related_params.vit_weights = DeepSeekVLV2VitWeight(
            {"vision": self.mm_part.vision, "projector": self.mm_part.projector}, True
        )
        # must use add_special_tokens=False
        self.image_id = self.tokenizer.tokenizer.encode(
            "<image>", add_special_tokens=False
        )[0]
        mm_related_params.special_token_ids.update(
            {"ignore_token_index": self.ignore_id, "image_token_index": self.image_id}
        )
        self.model_config.mm_model_config.mm_sep_tokens = [[self.image_id]]

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "gated-silu"
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            top_config_json = json.loads(content)
        DeepSeekVLV2._from_hf(config, top_config_json)
        DeepSeekVLV2._load_vit_param(config, top_config_json)
        return config

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        # Use GenericMoeModel with new config architecture
        # attention_type is determined from model_config.attn_config.use_mla
        self.py_model = GenericMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def _from_hf(config: ModelConfig, top_config_json: Dict[str, Any]):

        config.model_name = "deepseek_vl_v2"
        config_json = top_config_json["language_config"]
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.vocab_size = config_json["vocab_size"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.hidden_size = config_json["hidden_size"]
        assert config.hidden_size % config.attn_config.head_num == 0
        config.attn_config.size_per_head = (
            config.hidden_size // config.attn_config.head_num
        )
        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.attn_config.rope_config.style = 1

        # MLA config
        config.attn_config.use_mla = False

        # from Llama
        config.layernorm_eps = config_json.get(
            "rms_norm_eps", config_json.get("layer_norm_eps", 1e-05)
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
        candidate_resolutions = top_config_json.get("candidate_resolutions", {})
        config.mm_related_params.config["candidate_resolutions"] = candidate_resolutions
        config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})
        config.mm_related_params.config["tile_tag"] = top_config_json.get(
            "tile_tag", "2D"
        )
        config.mm_related_params.config["global_view_pos"] = top_config_json.get(
            "global_view_pos", "head"
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekVLV2Weight


register_model("deepseek_vl_v2", DeepSeekVLV2, ["DeepseekVL2ForCausalLM"])
