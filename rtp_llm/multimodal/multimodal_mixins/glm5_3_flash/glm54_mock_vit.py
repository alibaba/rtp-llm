import json
from pathlib import Path

import torch.nn.functional as F

from rtp_llm.multimodal.glm54_mock_vit_input import (
    GLM54_MOCK_IMAGE_TOKEN,
    UNSUPPORTED_IMAGE_OPTIONS,
    reject_image_option,
    validate_image_url,
    validate_pixel_bounds,
)
from rtp_llm.multimodal.mm_error_messages import raise_mm
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType

from .glm5_3_flash_mixin import (
    Glm53FlashImageEmbedding,
    Glm53FlashMixin,
    Glm53FlashVitWeight,
)


def configure_glm54_mock_vit(config, checkpoint_config, checkpoint_path):
    vision = dict(checkpoint_config["vision_config"])
    if vision["model_type"] != "glm5_next_vision":
        raise ValueError("GLM5.4 mock ViT requires GLM-5.3-Flash vision weights")
    # Match the Flash reference's vision block and post-attention norm epsilon.
    vision["rms_norm_eps"] = 1e-6
    image_start = int(checkpoint_config["image_start_token_id"])
    image_end = int(checkpoint_config["image_end_token_id"])
    with (Path(checkpoint_path) / "processor_config.json").open(encoding="utf-8") as f:
        processor = json.load(f)

    config.mm_model_config.is_multimodal = True
    config.mm_model_config.mm_sep_tokens = [[image_start, image_end]]
    config.mm_model_config.mm_position_ids_style = 0
    config.mm_related_params.special_tokens["default_mm_token"] = GLM54_MOCK_IMAGE_TOKEN
    config.mm_related_params.config.update(
        {
            "ckpt_path": checkpoint_path,
            "vision_config": vision,
            "processor_config": processor,
            "swiglu_limit": float(vision["swiglu_limit"]),
            "vision_special_token_ids": {
                "image_start": image_start,
                "image_end": image_end,
            },
            "language_hidden_size": config.hidden_size,
        }
    )


class Glm54MockImageEmbedding(Glm53FlashImageEmbedding):
    def __init__(self, mm_related_params):
        super().__init__(mm_related_params)
        self.padding = (
            mm_related_params.config["language_hidden_size"]
            - self.visual.config.out_hidden_size
        )
        if self.padding < 0:
            raise ValueError("GLM5.4 mock ViT output exceeds the language hidden size")

    @staticmethod
    def _validate_image_input(item: MultimodalInput) -> None:
        if item.mm_type not in (MMUrlType.DEFAULT, MMUrlType.IMAGE):
            raise_mm("GLM5.4 mock ViT accepts image inputs only")
        validate_image_url(item.url)
        config = item.mm_preprocess_config
        for api_name, native_name in UNSUPPORTED_IMAGE_OPTIONS:
            if getattr(config, native_name) != -1:
                reject_image_option(api_name)
        if config.crop_positions:
            reject_image_option("crop_positions")
        validate_pixel_bounds(config.min_pixels, config.max_pixels)

    def validate_inputs(self, mm_inputs: list[MultimodalInput]) -> None:
        for item in mm_inputs:
            self._validate_image_input(item)

    @staticmethod
    def preprocess_input(
        mm_inputs, vit_config, processor, video_processor, processor_config
    ):
        if len(mm_inputs) != 1:
            raise_mm("GLM5.4 mock ViT preprocessing expects one image at a time")
        Glm54MockImageEmbedding._validate_image_input(mm_inputs[0])
        return Glm53FlashImageEmbedding.preprocess_input(
            mm_inputs, vit_config, processor, video_processor, processor_config
        )

    def _format_embeddings(self, embeddings, data):
        # There is no learned cross-model projection in the mock checkpoint.
        return [F.pad(embeddings, (0, self.padding))], None, None


class Glm54MockVitMixin(Glm53FlashMixin):
    def _init_multimodal(self):
        self.mm_part = Glm54MockImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Glm53FlashVitWeight(
            {"vit": self.mm_part.visual}
        )


register_multimodal_mixin("glm_5", Glm54MockVitMixin)
