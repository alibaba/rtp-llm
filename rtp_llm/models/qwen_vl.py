import json
import os
from typing import Any, List

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
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = QwenVLImageEmbedding(self.model_config.mm_related_params)
        self.model_config.mm_related_params.vit_weights = QwenVLVitWeight(
            {"vit": self.mm_part.vit}
        )

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.max_seq_len = 1024
        config.vocab_size = 0
        QWen_VL._common_config(config, ckpt_path)
        return config

    @staticmethod
    def _common_config(config: ModelConfig, ckpt_path: str) -> ModelConfig:
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
        config.mm_related_params.eval_param_count = QWen_VL.eval_vit_param_count
        config.mm_related_params.eval_model_size = QWen_VL.eval_vit_model_size
        config.mm_model_config.mm_sep_tokens = [
            [vit_config["image_start_id"], vit_config["image_start_id"] + 1]
        ]

    @staticmethod
    def get_weight_cls():
        return QWenVLWeightInfo

    @classmethod
    def eval_vit_model_size(cls, mm_related_params):
        data_width = 4
        return QWen_VL.eval_vit_param_count(mm_related_params) * data_width

    @classmethod
    def eval_vit_param_count(cls, mm_related_params):
        vit_config = mm_related_params.config
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


register_model("qwen_vl", QWen_VL, ["QWenMLMHeadModel"])
