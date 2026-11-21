"""Kimi-K2.5 preprocessing wrapper; runtime classes live with NewLoader owners."""

import logging
from typing import List, Optional

import torch
from PIL import Image

from rtp_llm.models_py.new_models.kimi_k25.vision_runtime import *  # noqa: F401,F403
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)

_DEFAULT_MEDIA_PROC_CFG = {
    "in_patch_limit": 16384,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 4096,
    "in_patch_limit_video": None,
    "sample_fps": 2.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


class KimiK25ImageEmbedding(ImageEmbeddingInterface):
    """Top-level ViT + projector wrapped as MultiModalEmbeddingInterface."""

    def __init__(self, mm_related_params, vision_tower=None, mm_projector=None) -> None:
        cfg_dict = mm_related_params.config or {}
        vision_cfg_raw = cfg_dict.get("vision_config", {}) or {}
        self.vision_config = KimiK25VisionConfig(**vision_cfg_raw)
        # Attribute names match HF ckpt prefixes (`vision_tower.*`,
        # `mm_projector.*`) so weight loading via BaseVitWeights produces
        # correct ckpt key strings.
        self.vision_tower = (
            vision_tower
            if vision_tower is not None
            else MoonViT3dPretrainedModel(self.vision_config)
        )
        self.mm_projector = (
            mm_projector
            if mm_projector is not None
            else KimiK25MultiModalProjector(self.vision_config)
        )

        ckpt_path = cfg_dict.get("ckpt_path")
        self.image_processor = self._build_image_processor(ckpt_path)
        self.media_token_id = int(
            mm_related_params.special_token_ids.get("image_token_index", 163605)
        )

    @staticmethod
    def _build_image_processor(ckpt_path: Optional[str]):
        """Load the HF-side image processor, falling back to a built-in
        instance when ``trust_remote_code`` cannot resolve the ckpt path."""

        if ckpt_path is None:
            from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_image_processor import (
                KimiK25VisionProcessor,
            )

            return KimiK25VisionProcessor(media_proc_cfg=_DEFAULT_MEDIA_PROC_CFG)

        try:
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - depends on HF cache
            logging.warning(
                f"AutoImageProcessor.from_pretrained({ckpt_path}) failed ({exc}); "
                f"falling back to local KimiK25VisionProcessor"
            )
            from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_image_processor import (
                KimiK25VisionProcessor,
            )

            return KimiK25VisionProcessor(media_proc_cfg=_DEFAULT_MEDIA_PROC_CFG)

    @property
    def _device(self):
        return self.vision_tower.patch_embed.proj.weight.device

    @property
    def _data_type(self):
        # Source of truth is the vision tower's first conv weight dtype.
        # Used by image_embedding (line ~664) to cast pixel_values.
        return self.vision_tower.patch_embed.proj.weight.dtype

    def media_tokens_for(self, image: Image.Image) -> int:
        """Compute the number of mm_projector output tokens for an image
        without running the ViT forward."""
        media = {"type": "image", "image": image}
        return int(self.image_processor.media_tokens_calculator(media))

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image]) -> List[torch.Tensor]:
        # Returns a list of per-image (tokens, hidden) tensors so the base
        # `ImageEmbeddingInterface.mm_process` (`image_embedding([img])[0]`)
        # yields the full token sequence for that image rather than its
        # first row.
        device = self._device
        dtype = self._data_type

        medias = [{"type": "image", "image": img} for img in images]
        proc = self.image_processor.preprocess(medias, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device).to(dtype)
        grid_thws = proc["grid_thws"].to(device)

        vt_outputs = self.vision_tower(pixel_values, grid_thws)
        return mm_projector_forward(self.mm_projector, vt_outputs)

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        # MultiModalEmbeddingInterface contract: return (image_features, pos_ids).
        # Kimi K25 is not an mrope model so pos_ids is None. preprocess_input
        # delivers a single PIL.Image (see base ImageEmbeddingInterface.preprocess_input),
        # so we wrap into a list and unpack the [0] result from image_embedding.
        return self.image_embedding([data])[0], None
