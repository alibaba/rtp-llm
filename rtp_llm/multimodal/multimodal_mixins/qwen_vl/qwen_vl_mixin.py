from typing import List

import torch
from PIL import Image

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
    MultimodalInput,
    get_bytes_io_from_url,
)
from rtp_llm.multimodal.multimodal_mixins.qwen_vl.qwen_vl_vit import (
    VisionTransformer as QWen_VL_ViT,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class QwenVLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.visual."
        self._ft_prefix = "self.mm_part.vit."


class QwenVLImageEmbedding(ImageEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        self.vit = QWen_VL_ViT(**mm_related_params.config)
        self.mm_related_params = mm_related_params

    @property
    def _data_type(self):
        return self.vit.dtype

    @property
    def _device(self):
        return self.vit.device

    @staticmethod
    def preprocess_input(mm_inputs: List[MultimodalInput], vit_config: VitConfig):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        image = Image.open(data)
        return image

    @torch.inference_mode()
    def embedding(self, data, mm_type: MMUrlType, **kwargs):
        return self.vit.encode([data], self._device, self._data_type)[0], None


class QwenVLMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = QwenVLImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = QwenVLVitWeight({"vit": self.mm_part.vit})

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return QwenVLImageEmbedding(mm_related_params).vit


register_multimodal_mixin(["qwen_vl"], QwenVLMixin)
