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
        return self._project_images(data, pack_equal_rows=False)

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        if (
            getattr(self.embedding, "__func__", None)
            is not VisionBertEmbedding.embedding
        ):
            return super().batched_embedding(data_list, mm_types, **kwargs)
        return [
            self._project_images(data, pack_equal_rows=True)
            for data, _ in zip(data_list, mm_types)
        ]

    def _project_images(self, data, *, pack_equal_rows: bool):
        single = len(data) == 1 and data[0].ndim == 2 and data[0].is_contiguous()
        vision_input = data[0] if single else torch.cat(data, dim=0)
        weight = self.vision_embedding.weight
        vision_input = vision_input.to(
            dtype=weight.dtype, device=weight.device, non_blocking=True
        )
        vision_embedding = self.vision_embedding(vision_input)
        vision_embedding = self.LayerNorm(vision_embedding)
        if single:
            return [vision_embedding], None
        lengths = [tensor.size(0) for tensor in data]
        # The scheduler can retain one tensor for a uniform multi-image request.
        if (
            pack_equal_rows
            and len(lengths) > 1
            and lengths[0] > 0
            and lengths.count(lengths[0]) == len(lengths)
            and vision_embedding.ndim == 2
        ):
            return vision_embedding.unflatten(0, (len(lengths), lengths[0])), None
        return list(vision_embedding.split(lengths)), None


class VisionBertMixin(BaseMultiModalMixin):
    def _read_mm_weights(self, device: str):
        return self.mm_mixin_loader.load_weights(
            device=device, data_type=self.compute_dtype
        )

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
