"""DiffusionGemma multimodal mixin.

Vision path (HF reference: modeling_gemma4.py + modeling_diffusion_gemma.py):
  image -> Gemma4ImageProcessor -> (pixel_values [1,P,768], image_position_ids [1,P,2])
       -> Gemma4VisionModel (27 layers, axial 2D rope, qk_norm, MHA scaling=1.0)
       -> standardized soft tokens [S, 1152]
       -> DiffusionGemmaMultimodalEmbedder (scale-free RMSNorm + projection)
       -> features [S, text_hidden] spliced at image_token positions.

Checkpoint layout (two prefixes, both under model.encoder.):
  model.encoder.vision_tower.<key>  -> mm_part.visual.<key>
  model.encoder.embed_vision.<key>   -> mm_part.embed_vision.<key>
"""

import logging
from typing import Any, List, Optional

import torch
import torch.library as tl
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Gemma4VisionModel
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaMultimodalEmbedder,
)

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.flash_attn_utils import can_use_flash_attn
from rtp_llm.utils.model_weight import CkptWeightInfo, identity

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton

default_attn_impl = "sdpa"
try:
    if can_use_flash_attn():
        default_attn_impl = "flash_attention_2"
except Exception as e:
    logging.info(
        f"initialize flash_attn failed, exception {e}, using sdpa attention in diffusion_gemma vit"
    )


class DiffusionGemmaImageEmbedding:
    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]
        self.mm_processor = AutoProcessor.from_pretrained(ckpt_path)
        config_hf = AutoConfig.from_pretrained(ckpt_path)
        config_hf.vision_config._attn_implementation = default_attn_impl
        self.visual = Gemma4VisionModel._from_config(config_hf.vision_config)
        self.embed_vision = DiffusionGemmaMultimodalEmbedder(
            config_hf.vision_config, config_hf.text_config
        )

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config,
        processor,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        if mm_type != MMUrlType.DEFAULT and mm_type != MMUrlType.IMAGE:
            raise Exception("diffusion_gemma only supports image inputs")
        image_data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        image = Image.open(image_data).convert("RGB")
        res = processor.image_processor(image, return_tensors="pt")
        return res["pixel_values"], res["image_position_ids"]

    def get_preprocess_params(self):
        return {"processor": self.mm_processor}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        position_ids = data[1].to(self._device)
        vision_output = self.visual(
            pixel_values=pixel_values, pixel_position_ids=position_ids
        )
        features = self.embed_vision(vision_output.last_hidden_state)
        return features, None


class DiffusionGemmaVitWeight(BaseVitWeights):
    """Two modules under two checkpoint prefixes; the deploy weight info below
    maps each weight name to its own checkpoint key."""

    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.encoder."
        self._ft_prefix = "self.mm_part."

    @staticmethod
    def ckpt_key(weight_name: str) -> str:
        if weight_name.startswith("visual."):
            return "model.encoder.vision_tower." + weight_name[len("visual.") :]
        return "model.encoder." + weight_name


class DiffusionGemmaVitDeployWeightInfo(BaseMultiModalDeployWeightInfo):
    def get_weight_info(self):
        weights = []
        for w in self.vit_weights.weight_names:
            weights.append(
                CustomAtomicWeight(
                    w,
                    [CkptWeightInfo(self.vit_weights.ckpt_key(w), identity)],
                    identity,
                )
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class DiffusionGemmaMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = DiffusionGemmaImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = DiffusionGemmaVitWeight(
            {"visual": self.mm_part.visual, "embed_vision": self.mm_part.embed_vision}
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls):
        return DiffusionGemmaVitDeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config):
        mm_part = DiffusionGemmaImageEmbedding(mm_related_params)
        return mm_part.visual


register_multimodal_mixin(["diffusion_gemma"], DiffusionGemmaMixin)
