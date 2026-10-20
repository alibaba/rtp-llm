"""DeepSeek V4.1 vision integration with the common mixin and scheduler."""

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.multimodal_mixin_loader import MultimodalMixinLoader
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)

from .deepseek_v41_vision import DeepSeekV41VisionEmbedding


class V41VitWeights(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."


class V41MixinLoader(MultimodalMixinLoader):
    def __init__(
        self,
        weights_info,
        database,
        load_method=LoadMethod.AUTO,
        *,
        data_type=torch.float32,
    ):
        super().__init__(weights_info, database, load_method)
        self._data_type = data_type

    def load_weights(self, device="cpu", data_type=None):
        if data_type is None:
            data_type = self._data_type
        return super().load_weights(device=device, data_type=data_type)


class DeepSeekV41Mixin(BaseMultiModalMixin):
    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        config = mm_related_params.config["v41_config"]
        if not isinstance(config, V41Config):
            config = V41Config.from_dict(config)
        return DeepSeekV41VisionEmbedding(
            config,
            max_batch_images=vit_config.embedding_scheduler_args()["max_batch_images"],
        )

    def _init_multimodal(self):
        self.mm_part = self._get_mm_module(self.mm_related_params, self.vit_config)
        self.mm_related_params.vit_weights = V41VitWeights(
            {"vision_model": self.mm_part}
        )

    def create_mm_mixin_loader(self):
        loader = super().create_mm_mixin_loader()
        return V41MixinLoader(
            loader.weights_info,
            loader.database,
            loader.load_method,
            data_type=self.mm_part.vision.norm.weight.dtype,
        )

    def load_mm_weight(self, ctype, device):
        if not self.weights:
            raise RuntimeError(f"No V4.1 vision weights loaded from {self.ckpt_path!r}")
        state = self.mm_part.state_dict()
        state = {
            name: self.weights[name]
            .reshape(tensor.shape)
            .to(device=device, dtype=tensor.dtype)
            for name, tensor in state.items()
        }
        self.mm_part.load_state_dict(state, strict=True, assign=True)
        self.mm_part.eval().requires_grad_(False)
        self.weights.clear()


register_multimodal_mixin(["deepseek_v41", "deepseek_v41_dspark"], DeepSeekV41Mixin)
