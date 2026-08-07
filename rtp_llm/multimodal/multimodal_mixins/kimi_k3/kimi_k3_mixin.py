"""Kimi-K3 multimodal mixin: wires the MoonViT 3D + projector into the
new ``BaseMultiModalMixin`` / mixin-factory pattern.

The actual ViT (vision tower + projector) lives in :mod:`kimi_k3_vit`.
This file just bridges it to the runtime via ``KimiK3Mixin``, which is
the class the :class:`MultimodalMixinFactory` instantiates when
``model_config.model_type == "kimi_k3"``.
"""

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.utils.base_model_datatypes import VitParameters


class KimiK3VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        # HF Kimi-K3 keeps vision_tower.* / mm_projector.* at top level;
        # no `model.` / `language_model.` prefix.
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."


class KimiK3Mixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = KimiK3ImageEmbedding(self.mm_related_params)
        # vit_weights dict keys must match the on-disk ckpt prefix segment so
        # that BaseVitWeights builds names like `vision_tower.*` / `mm_projector.*`.
        self.mm_related_params.vit_weights = KimiK3VitWeight(
            {
                "vision_tower": self.mm_part.vision_tower,
                "mm_projector": self.mm_part.mm_projector,
            },
            with_prefix=True,
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        mm_part = KimiK3ImageEmbedding(mm_related_params)
        return torch.nn.ModuleList(
            [
                mm_part.vision_tower,
                mm_part.mm_projector,
            ]
        )


register_multimodal_mixin(["kimi_k3"], KimiK3Mixin)
