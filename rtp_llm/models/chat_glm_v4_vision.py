import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.chat_glm_v4 import ChatGlmV4
from rtp_llm.models.chat_glm_v4_vision_weight import (
    ChatGlmV4VisionVitWeights,
    ChatGlmV4VisionWeightInfo,
)
from rtp_llm.models.eva2clip_vit import EVA2CLIPImageEmbedding
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.util import get_config_from_path


class ChatGlmV4VisionImageEmbedding(EVA2CLIPImageEmbedding):
    @torch.inference_mode()
    def embedding(self, data, mm_type: MMUrlType, **kwargs):
        tensor_images = self.image_transform.encode(
            [data], self._device, self._data_type
        )
        tensor_images = self.vit(tensor_images).to(device=self._device)[0]
        pos_ids = torch.ones(tensor_images.shape[0], dtype=torch.int32)
        pos_ids[0] = 0
        pos_ids[-1] = 2
        return tensor_images, pos_ids


class ChatGlmV4Vision(ChatGlmV4, MultiModalMixin):
    def _init_multimodal(self):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = ChatGlmV4VisionImageEmbedding(
            self.model_config.mm_related_params
        )
        self.model_config.mm_related_params.vit_weights = ChatGlmV4VisionVitWeights(
            {"vit": self.mm_part.vit}
        )

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return ChatGlmV4VisionImageEmbedding(config).vit

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ChatGlmV4._create_config(ckpt_path)
        config_dict = get_config_from_path(ckpt_path)
        vit_config = config_dict["vision_config"]
        config.mm_related_params.config.update(vit_config)
        # use initial hidden size for linear_proj and conv layer in eva2clip
        config.mm_related_params.config["use_vision_hidden_size"] = False
        config.mm_related_params.config["boi_token_id"] = config_dict.get(
            "boi_token_id", 0
        )
        config.mm_related_params.config["eoi_token_id"] = config_dict.get(
            "eoi_token_id", 0
        )
        config.mm_model_config.mm_sep_tokens = [
            [config_dict.get("boi_token_id", 0), config_dict.get("eoi_token_id", 0)]
        ]
        config.include_sep_tokens = True
        config.mm_position_ids_style = 1
        return config

    @staticmethod
    def get_weight_cls():
        return ChatGlmV4VisionWeightInfo


register_model("chatglm4v", ChatGlmV4Vision, [], ["THUDM/glm-4v-9b"])
