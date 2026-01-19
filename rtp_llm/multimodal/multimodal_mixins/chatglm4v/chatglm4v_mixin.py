import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.chatglm4v.eva2clip_vit import (
    EVA2CLIPImageEmbedding,
)
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


class ChatGlmV4VisionVitWeights(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.vision."
        self._ft_prefix = "self.mm_part.vit."


class ChatGlmV4VisionMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = ChatGlmV4VisionImageEmbedding(self.model_config)
        self.model_config.mm_related_params.vit_weights = ChatGlmV4VisionVitWeights(
            {"vit": self.mm_part.vit}
        )

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return ChatGlmV4VisionImageEmbedding(config).vit


register_multimodal_mixin(["chatglm4v"], ChatGlmV4VisionMixin)
