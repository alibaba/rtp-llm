"""Kimi-K2.5 multimodal mixin: wires the MoonViT 3D + projector into the
new ``BaseMultiModalMixin`` / mixin-factory pattern.

The actual ViT (vision tower + projector) lives in :mod:`kimi_k25_vit`.
This file just bridges it to the runtime via ``KimiK25Mixin``, which is
the class the :class:`MultimodalMixinFactory` instantiates when
``model_config.model_type == "kimi_k25"``.
"""

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_vit import (
    KimiK25ImageEmbedding,
)
from rtp_llm.utils.base_model_datatypes import VitParameters


class KimiK25Mixin(BaseMultiModalMixin):
    def __init__(
        self,
        compute_dtype,
        device,
        mm_related_params,
        load_method,
        vit_config,
        ckpt_path,
    ):
        from rtp_llm.models_py.new_models.kimi_k25.vision import load_kimi_k25_vision

        owner = load_kimi_k25_vision(
            mm_related_params.config["vision_config"],
            ckpt_path,
            compute_dtype,
            device,
            load_method,
        )
        self.compute_dtype = compute_dtype
        self.mm_related_params = mm_related_params
        self.load_method = load_method
        self.vit_config = vit_config
        self.ckpt_path = ckpt_path
        self.mm_part = KimiK25ImageEmbedding(
            mm_related_params,
            vision_tower=owner.vision_tower,
            mm_projector=owner.mm_projector,
        )

    @classmethod
    def from_new_loader(
        cls,
        compute_dtype,
        device,
        mm_related_params,
        load_method,
        vit_config,
        ckpt_path,
    ):
        """Public factory entry; both construction paths use direct checkpoint owners."""
        return cls(
            compute_dtype, device, mm_related_params, load_method, vit_config, ckpt_path
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        mm_part = KimiK25ImageEmbedding(mm_related_params)
        return torch.nn.ModuleList(
            [
                mm_part.vision_tower,
                mm_part.mm_projector,
            ]
        )


register_multimodal_mixin(["kimi_k25"], KimiK25Mixin)
