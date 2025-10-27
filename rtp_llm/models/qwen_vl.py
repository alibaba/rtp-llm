import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch
from transformers import AutoTokenizer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel, MultimodalInput
from rtp_llm.models.multimodal.multimodal_common import ImageEmbeddingInterface, mm_lock
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models.qwen import QWen
from rtp_llm.models.qwen_vl_vit import VisionTransformer as QWen_VL_ViT
from rtp_llm.models.qwen_vl_weight import QwenVLVitWeight, QWenVLWeightInfo
from rtp_llm.utils.base_model_datatypes import MMUrlType


class QwenVLImageEmbedding(ImageEmbeddingInterface):
    def __init__(self, config: GptInitModelParameters):
        self.vit = QWen_VL_ViT(**config.mm_related_params.config)

    @property
    def _device(self):
        return self.vit.device

    @torch.inference_mode()
    def embedding(self, data, mm_type: MMUrlType, **kwargs):
        return self.vit.encode(data, self._device, self._data_type), None


class QWen_VL(QWen, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = QwenVLImageEmbedding(config)
        config.mm_related_params.vit_weights = QwenVLVitWeight(
            {"vit": self.mm_part.vit}
        )

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0, size_per_head=0, layer_num=0, max_seq_len=1024, vocab_size=0
        )
        QWen_VL._common_config(config, ckpt_path)
        return config

    @staticmethod
    def _common_config(
        config: GptInitModelParameters, ckpt_path: str
    ) -> GptInitModelParameters:
        QWen._common_config(config, ckpt_path)
        QWen._from_hf(config, ckpt_path)
        QWen_VL._load_vit_param(config, ckpt_path)
        return config

    @staticmethod
    def _load_vit_param(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        vit_config = config_json["visual"]
        config.mm_related_params.config.update(vit_config)
        config.mm_related_params.special_token_ids.update(
            {
                "image_start_id": vit_config["image_start_id"],
                "image_end_id": vit_config["image_start_id"] + 1,
                "image_pad_id": vit_config["image_start_id"] + 2,
            }
        )
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_sep_tokens = [
            [vit_config["image_start_id"], vit_config["image_start_id"] + 1]
        ]

    @staticmethod
    def get_weight_cls():
        return QWenVLWeightInfo

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        llm_size = BaseModel.eval_model_size(config)

        data_width = 4
        llm_size += QWen_VL.eval_vit_param_count(config) * data_width
        return llm_size

    @staticmethod
    def eval_vit_param_count(config: GptInitModelParameters):
        vit_config = config.mm_related_params.config
        embed_dim = vit_config["output_dim"]
        width = vit_config["width"]
        layers = vit_config["layers"]
        patch_size = vit_config["patch_size"]
        mlp_ratio = vit_config["mlp_ratio"]
        mlp_width = int(mlp_ratio * width)

        llm_size = 3 * width * patch_size**2 + width * 2
        llm_size += layers * (
            width * 2 * 2
            + width**2 * 4
            + width * 4
            + mlp_width * width * 2
            + mlp_width
            + width
        )
        llm_size += width * embed_dim + embed_dim**2 + embed_dim + embed_dim * 2 * 3
        return llm_size

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        llm_param_count = BaseModel.eval_model_param_count(config)
        llm_param_count += QWen_VL.eval_vit_param_count(config)

        return llm_param_count


register_model("qwen_vl", QWen_VL, ["QWenMLMHeadModel"])
