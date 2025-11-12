import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_common import ImageEmbeddingInterface
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models.qwen import QWen
from rtp_llm.models.qwen_vl_vit import VisionTransformer as QWen_VL_ViT
from rtp_llm.models.qwen_vl_weight import QwenVLVitWeight, QWenVLWeightInfo


class QwenVLImageEmbedding(ImageEmbeddingInterface):
    def __init__(self, mm_related_params):
        if mm_related_params is None or mm_related_params.config is None:
            raise ValueError("mm_related_params.config is required for QwenVLImageEmbedding")
        self.vit = QWen_VL_ViT(**mm_related_params.config)
        self.mm_related_params = mm_related_params

    @property
    def _device(self):
        return self.vit.device

    @torch.no_grad()
    def image_embedding(self, images: List[Any]) -> torch.Tensor:
        images = self.vit.encode(images, self._device, self._data_type)
        return images


class QWen_VL(QWen, MultiModalMixin):
    def _init_multimodal(self, mm_model_config, vit_config):
        if mm_model_config.mm_related_params is None:
            raise ValueError("mm_model_config.mm_related_params is required for QWen_VL")
        self.mm_part = QwenVLImageEmbedding(mm_model_config.mm_related_params)
        mm_model_config.mm_related_params.vit_weights = QwenVLVitWeight(
            {"vit": self.mm_part.vit}
        )

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.config.model_config import ModelConfig
        config = ModelConfig()
        config.head_num_ = 0
        config.size_per_head_ = 0
        config.num_layers = 0
        config.max_seq_len = 1024
        config.vocab_size = 0
        QWen_VL._common_config(config, ckpt_path)
        return config

    @staticmethod
    def _common_config(
        config: ModelConfig, ckpt_path: str
    ) -> ModelConfig:
        QWen._common_config(config, ckpt_path)
        QWen._from_hf(config, ckpt_path)
        QWen_VL._load_vit_param(config, ckpt_path)
        return config

    @staticmethod
    def _load_vit_param(config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        from rtp_llm.config.model_config import VitParameters
        if config.mm_related_params is None:
            config.mm_related_params = VitParameters()
        vit_config = config_json["visual"]
        if config.mm_related_params.config is None:
            config.mm_related_params.config = {}
        config.mm_related_params.config.update(vit_config)
        if config.mm_related_params.special_token_ids is None:
            config.mm_related_params.special_token_ids = {}
        config.mm_related_params.special_token_ids.update(
            {
                "image_start_id": vit_config["image_start_id"],
                "image_end_id": vit_config["image_start_id"] + 1,
                "image_pad_id": vit_config["image_start_id"] + 2,
            }
        )
        if config.mm_related_params.special_tokens is None:
            config.mm_related_params.special_tokens = {}
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_sep_tokens = [
            [vit_config["image_start_id"], vit_config["image_start_id"] + 1]
        ]

    @staticmethod
    def get_weight_cls():
        return QWenVLWeightInfo

    @staticmethod
    def eval_model_size(config: ModelConfig):
        llm_size = config.eval_model_size()

        data_width = 4
        llm_size += QWen_VL.eval_vit_param_count(config) * data_width
        return llm_size

    @staticmethod
    def eval_vit_param_count(config: ModelConfig):
        if config.mm_related_params is None or config.mm_related_params.config is None:
            return 0
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
    def eval_model_param_count(config: ModelConfig):
        llm_param_count = config.model_param_count()
        llm_param_count += QWen_VL.eval_vit_param_count(config)

        return llm_param_count


register_model("qwen_vl", QWen_VL, ["QWenMLMHeadModel"])
