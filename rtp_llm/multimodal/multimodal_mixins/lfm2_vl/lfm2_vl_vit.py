"""LFM2-VL multimodal ViT: SigLIP2 NaFlex tower + projector.

The tower is instantiated from the pinned ``transformers`` SigLIP2 vision
classes (the same construction ``Lfm2VlModel`` itself uses:
``AutoModel.from_config(vision_config)``), so its module tree matches the
checkpoint names ``model.vision_tower.vision_model.*`` exactly.

The projector is a verbatim port of HF ``Lfm2VlMultiModalProjector``:
pixel_unshuffle(factor=2) -> linear_1 -> gelu -> linear_2 (both with bias),
keeping the HF transposition order so numerics match bit-for-bit.
"""

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from rtp_llm.multimodal.multimodal_mixins.lfm2_vl.lfm2_vl_image_processor import (
    Lfm2VlVisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
    get_bytes_io_from_url,
)

logger = logging.getLogger(__name__)


class Lfm2VlProjector(nn.Module):
    """HF Lfm2VlMultiModalProjector port (model.multi_modal_projector.*).

    ``config`` is the full top-level VL config dict (d1 config.json), which
    carries text_config/vision_config/projector_* fields.
    """

    def __init__(self, config: dict):
        super().__init__()
        text_hidden = int(config["text_config"]["hidden_size"])
        vision_hidden = int(config["vision_config"]["hidden_size"])
        self.factor = int(config.get("downsample_factor", 2))
        in_channels = vision_hidden * self.factor**2
        hidden = int(config.get("projector_hidden_size", 2048))
        self.use_layer_norm = bool(config.get("projector_use_layernorm", False))
        self.layer_norm = (
            nn.LayerNorm(in_channels) if self.use_layer_norm else nn.Identity()
        )
        # attribute names match checkpoint: multi_modal_projector.linear_1/2
        self.linear_1 = nn.Linear(in_channels, hidden, bias=True)
        self.linear_2 = nn.Linear(hidden, text_hidden, bias=True)

    def pixel_unshuffle(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # verbatim HF transposition order; input (B, H, W, C)
        batch_size, width, height, channels = hidden_states.size()
        hidden_states = hidden_states.reshape(
            batch_size, width, height // self.factor, channels * self.factor
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size,
            height // self.factor,
            width // self.factor,
            channels * self.factor**2,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.pixel_unshuffle(image_features)
        image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = F.gelu(hidden_states)
        return self.linear_2(hidden_states)


class Lfm2VlImageEmbedding(ImageEmbeddingInterface):
    """LFM2-VL tower + projector as MultiModalEmbeddingInterface."""

    def __init__(self, mm_related_params) -> None:
        cfg_dict = mm_related_params.config or {}
        ckpt_path = cfg_dict.get("ckpt_path")
        self.mm_config = self._load_hf_vl_config(ckpt_path)
        vision_config = self.mm_config.vision_config

        from transformers import AutoModel

        # same construction as HF Lfm2VlModel.__init__
        self.vision_tower = AutoModel.from_config(vision_config)
        self.multi_modal_projector = Lfm2VlProjector(cfg_dict.get("vl_config", {}))
        self.image_processor = Lfm2VlVisionProcessor(
            cfg_dict.get("image_processor_config", {})
        )

    @staticmethod
    def _load_hf_vl_config(ckpt_path: Optional[str]):
        from transformers import Lfm2VlConfig

        return Lfm2VlConfig.from_pretrained(ckpt_path)

    @property
    def _device(self):
        return self.vision_tower.device

    @property
    def _data_type(self):
        return self.vision_tower.dtype

    @staticmethod
    def preprocess_input(
        mm_inputs,
        vit_config,
        processor,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        image = Image.open(data).convert("RGB")
        res = processor(image, mm_preprocess_config=mm_input.mm_preprocess_config)
        return res["pixel_values"], res["spatial_shapes"], res["pixel_attention_mask"]

    def get_preprocess_params(self):
        return {"processor": self.image_processor}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        spatial_shapes = data[1].to(self._device)
        pixel_attention_mask = data[2].to(self._device)

        image_outputs = self.vision_tower(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
        )
        last_hidden_state = image_outputs.last_hidden_state

        img_feature_lengths = pixel_attention_mask.sum(dim=1)
        image_features: List[torch.Tensor] = []
        for img_idx in range(last_hidden_state.size(0)):
            feature = last_hidden_state[img_idx]
            feature = feature[: img_feature_lengths[img_idx], :].unsqueeze(0)
            feature_h, feature_w = spatial_shapes[img_idx].tolist()
            feature = feature.reshape(1, feature_h, feature_w, -1)
            img_embedding = self.multi_modal_projector(feature)
            image_features.append(img_embedding.reshape(-1, img_embedding.size(-1)))
        features = torch.cat(image_features, dim=0).to(self._data_type)
        # LFM2 uses plain sequential RoPE positions for image tokens
        return features, None
