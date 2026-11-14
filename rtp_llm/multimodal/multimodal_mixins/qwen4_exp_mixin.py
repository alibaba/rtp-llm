"""Thin Qwen4 adapter over the existing Qwen3.5 vision tower."""

import torch

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    Qwen3_5MoeMixin,
    Qwen3_5MoeVitWeight,
)
from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_processor import (
    assemble_visual_embedding,
    preprocess_qwen4,
)


class Qwen4ExpImageEmbedding(Qwen3_5MoeImageEmbedding):
    # Keep the checkpoint's official fast processor; the shared tower and
    # weight loader are the same as Qwen3.5.
    image_processor_class = None

    def __init__(self, mm_related_params):
        super().__init__(mm_related_params)
        if getattr(self.visual.config, "deepstack_visual_indexes", None):
            raise ValueError(
                "Qwen4 adapter requires the released non-DeepStack vision tower"
            )

    preprocess_input = staticmethod(preprocess_qwen4)

    def get_preprocess_params(self):
        return {"processor": self.mm_processor}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return self.batched_embedding([data], [None], **kwargs)[0]

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        results = self._batched_vision_embedding(
            [(data.pixels, data.grid) for data in data_list], **kwargs
        )
        return [
            assemble_visual_embedding(
                data, features, positions, self.mm_processor, self.text_embedding
            )
            for data, (features, positions) in zip(data_list, results)
        ]


class Qwen4ExpMixin(Qwen3_5MoeMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen4ExpImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )

    def load_mm_weight(self, ctype, device):
        super().load_mm_weight(ctype, device)
        database = self.mm_mixin_loader.database
        for name in (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ):
            if database.has_tensor(name):
                self.mm_part.text_embedding = database.load_tensor(
                    name, data_type=self.compute_dtype
                )[0].cpu()
                break
        else:
            raise ValueError("Qwen4 checkpoint is missing its text embedding weight")

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        return Qwen4ExpImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen4_exp"], Qwen4ExpMixin)
