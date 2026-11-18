from typing import List

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.vision_bert.configuration_mbert import MBertRelConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import VitParameters
from rtp_llm.utils.util import get_config_from_path


class VisionBertEmbedding(ImageEmbeddingInterface):
    # Inputs are ready-to-project feature tensors, so preprocessing only
    # unpacks them before the projection and normalization.
    supports_inline_embedding = True

    def __init__(self, mm_related_params: VitParameters):
        config_json = get_config_from_path(mm_related_params.config["ckpt_path"])
        if config_json is None:
            raise Exception(
                "failed to get config json from path: "
                + mm_related_params.config["ckpt_path"]
            )
        # config
        config: MBertRelConfig = MBertRelConfig.from_dict(config_json)
        self.hidden_size: int = config.hidden_size
        self.vision_seq_len: int = config.vision_seq_len
        if self.vision_seq_len < 2:
            raise Exception("vision seq len not support less than 2 currently")
        self.vision_hidden_size: int = config.vision_hidden_size
        self.vision_embedding_opt = config.vision_embedding_opt
        # network
        self.vision_embedding = torch.nn.Linear(
            self.vision_hidden_size, self.hidden_size
        )
        if not self.vision_embedding_opt:
            raise Exception("not support vision_embedding_opt = false")
        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    @property
    def _data_type(self):
        return self.vision_embedding.weight.dtype

    @property
    def _device(self):
        return self.vision_embedding.weight.device

    @staticmethod
    def preprocess_input(mm_inputs: List[MultimodalInput], vit_config, **kwargs):
        return [mm_input.tensor for mm_input in mm_inputs]

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        tensors = data
        input_length = [len(t) for t in tensors]
        vision_input = torch.concat(tensors, dim=0).to(
            dtype=self._data_type, device=self._device, non_blocking=True
        )

        vision_embedding = self.vision_embedding(vision_input)
        vision_embedding = self.LayerNorm(vision_embedding)
        return list(torch.split(vision_embedding, input_length, dim=0)), None


class VisionBertMixin(BaseMultiModalMixin):
    def _init_multimodal(
        self,
    ) -> None:
        self.mm_part = VisionBertEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = BaseVitWeights(
            {
                "vision_embedding": self.mm_part.vision_embedding,
                "LayerNorm": self.mm_part.LayerNorm,
            },
            with_prefix=True,
        )
        self.mm_related_params.vit_weights._ckpt_prefix = "vision_embedding."

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        embed = VisionBertEmbedding(mm_related_params)
        return torch.nn.ModuleList([embed.vision_embedding, embed.LayerNorm])


register_multimodal_mixin("vision_bert", VisionBertMixin)
