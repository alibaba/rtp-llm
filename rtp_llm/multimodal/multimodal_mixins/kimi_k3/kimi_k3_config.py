"""Checkpoint-derived K3 image configuration shared with the text model loader."""

from typing import TYPE_CHECKING, Any, Dict

from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    load_kimi_k3_media_config,
)

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig


def configure_kimi_k3_multimodal(
    config: "ModelConfig", top_config: Dict[str, Any]
) -> None:
    """Bind one media placeholder to one complete MoonViT feature sequence."""
    if "vision_config" not in top_config:
        config.mm_model_config.is_multimodal = False
        return

    vision_config = dict(top_config["vision_config"])
    vision_config.pop("_name_or_path", None)
    if "media_placeholder_token_id" not in top_config:
        raise ValueError("Kimi K3 vision config has no media_placeholder_token_id")
    image_placeholder = top_config.get("image_placeholder")
    if not isinstance(image_placeholder, str) or not image_placeholder:
        raise ValueError("Kimi K3 vision config has no image_placeholder")
    media_token_id = int(top_config["media_placeholder_token_id"])

    config.mm_model_config.is_multimodal = True
    config.mm_model_config.mm_sep_tokens = [[media_token_id]]
    config.mm_related_params.config = {
        "vision_config": vision_config,
        "media_proc_cfg": load_kimi_k3_media_config(config.ckpt_path),
        "ckpt_path": config.ckpt_path,
    }
    config.mm_related_params.special_token_ids = dict(
        config.mm_related_params.special_token_ids
    )
    config.mm_related_params.special_tokens = dict(config.mm_related_params.special_tokens)
    config.mm_related_params.special_token_ids["image_token_index"] = media_token_id
    config.mm_related_params.special_tokens.update(
        {
            "default_mm_token": "<|media_pad|>",
            "image_placeholder": image_placeholder,
        }
    )
    config.mm_related_params.support_batch = True
