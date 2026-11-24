"""DeepSeek V4.1 vision integration with the common mixin and scheduler."""

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
)

from .deepseek_v41_vision import DeepSeekV41VisionEmbedding, load_v41_service_vision


class DeepSeekV41Mixin(BaseMultiModalMixin):
    def __init__(
        self,
        compute_dtype,
        device,
        mm_related_params,
        load_method,
        vit_config,
        ckpt_path,
    ):
        # Bypass the common base's legacy descriptor/intermediate-dict path.
        self.compute_dtype = compute_dtype
        self.mm_related_params = mm_related_params
        self.vit_config = vit_config
        self.load_method = load_method
        self.ckpt_path = ckpt_path
        config = mm_related_params.config["v41_config"]
        if not isinstance(config, V41Config):
            config = V41Config.from_dict(config)
        owner, loader = load_v41_service_vision(
            config, ckpt_path, compute_dtype, device, load_method
        )
        self.mm_part = self.checkpoint_owner = owner
        self.model_weights_loader = loader
        self.mm_related_params.vit_weights = None

    @classmethod
    def from_new_loader(
        cls, compute_dtype, device, mm_related_params, load_method, vit_config, ckpt_path
    ):
        # Both entry points publish the same validated V4.1 vision owner.
        return cls(
            compute_dtype, device, mm_related_params, load_method, vit_config, ckpt_path
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        config = mm_related_params.config["v41_config"]
        if not isinstance(config, V41Config):
            config = V41Config.from_dict(config)
        return DeepSeekV41VisionEmbedding(config)


register_multimodal_mixin(["deepseek_v41", "deepseek_v41_dspark"], DeepSeekV41Mixin)
