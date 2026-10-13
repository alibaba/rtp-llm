"""LFM2-VL multimodal mixin: wires the SigLIP2 NaFlex tower + projector into
the ``BaseMultiModalMixin`` pattern (kimi_k25-style).

The actual tower/projector live in :mod:`lfm2_vl_vit`; the image
preprocessing port in :mod:`lfm2_vl_image_processor`.
"""

import torch

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.lfm2_vl.lfm2_vl_vit import (
    Lfm2VlImageEmbedding,
)
from rtp_llm.utils.base_model_datatypes import VitParameters


class Lfm2VlVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        # d1 checkpoint: model.vision_tower.vision_model.* / model.multi_modal_projector.*
        self._ckpt_prefix = "model."
        self._ft_prefix = "self.mm_part."


class Lfm2VlMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Lfm2VlImageEmbedding(self.mm_related_params)
        # vit_weights dict keys must match the on-disk ckpt prefix segments so
        # BaseVitWeights builds names like `vision_tower.vision_model...` and
        # `multi_modal_projector.linear_1.weight`.
        self.mm_related_params.vit_weights = Lfm2VlVitWeight(
            {
                "vision_tower": self.mm_part.vision_tower,
                "multi_modal_projector": self.mm_part.multi_modal_projector,
            },
            with_prefix=True,
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config):
        mm_part = Lfm2VlImageEmbedding(mm_related_params)
        return torch.nn.ModuleList(
            [
                mm_part.vision_tower,
                mm_part.multi_modal_projector,
            ]
        )


register_multimodal_mixin(["lfm2_vl"], Lfm2VlMixin)
