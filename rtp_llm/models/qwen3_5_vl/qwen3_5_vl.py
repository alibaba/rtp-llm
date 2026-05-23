import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen2_5_vl.qwen2_5_vl import Qwen2_5_VLImageEmbedding
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35MoeWeight
from rtp_llm.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3_VL_MOEVisionTransformerPretrainedModel,
)


class Qwen35VLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class Qwen35VLImageEmbedding(Qwen2_5_VLImageEmbedding):
    def __init__(self, mm_related_params, model_config=None):
        super().__init__(mm_related_params, model_config=model_config)
        self.visual = Qwen3_VL_MOEVisionTransformerPretrainedModel(
            mm_related_params.config
        )


class Qwen35VLMoeWeight(Qwen35MoeWeight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        Qwen35MoeWeight.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)


class Qwen35VLMoe(Qwen35Moe, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config: Any,
        vit_config: VitConfig,
    ):
        mm_related_params = self.model_config.mm_related_params
        self.mm_part = Qwen35VLImageEmbedding(
            mm_related_params, model_config=self.model_config
        )
        self.model_config.mm_related_params.vit_weights = Qwen35VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @staticmethod
    def get_weight_cls():
        return Qwen35VLMoeWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen35_vl_next import Qwen35VLNextModel

        self.py_model = Qwen35VLNextModel(
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

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        cls._load_vit_param(config, config_json, ckpt_path)
        cls._load_mrope_param(config, config_json["text_config"])
        return config

    @staticmethod
    def _load_vit_param(
        config: ModelConfig, config_json: Dict[str, Any], ckpt_path: str
    ):
        config.mm_related_params.config = dict(config_json["vision_config"])
        config.mm_related_params.config["ckpt_path"] = ckpt_path
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]
        config.mm_model_config.mm_position_ids_style = 2

    @staticmethod
    def _load_mrope_param(config: ModelConfig, text_config_json: Dict[str, Any]):
        rope_parameters = text_config_json["rope_parameters"]
        mrope_section = rope_parameters["mrope_section"]
        rope_config = config.attn_config.rope_config
        rope_config.style = 7
        rope_config.base = rope_parameters["rope_theta"]
        rope_config.index_factor = len(mrope_section)
        rope_config.mrope_dim1 = mrope_section[0]
        rope_config.mrope_dim2 = mrope_section[1]
        rope_config.mrope_dim3 = mrope_section[2]
        rope_config.dim = sum(mrope_section)


register_model("qwen35_vl_moe", Qwen35VLMoe)
