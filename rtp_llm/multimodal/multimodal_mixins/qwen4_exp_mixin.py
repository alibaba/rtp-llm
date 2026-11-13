"""Thin Qwen4 adapter over the existing Qwen3.5 vision tower."""

from dataclasses import replace

import torch
from transformers import AutoProcessor

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    Qwen3_5MoeMixin,
    Qwen3_5MoeVitWeight,
)
from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_processor import (
    assemble_visual_embedding,
    preprocess_qwen4,
    qwen4_token_length,
)


class Qwen4ExpImageEmbedding(Qwen3_5MoeImageEmbedding):
    # Keep the checkpoint's official fast processor; the shared tower and
    # weight loader are the same as Qwen3.5.
    image_processor_class = None

    @staticmethod
    def _load_processor(ckpt_path):
        return AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)

    def __init__(self, mm_related_params):
        super().__init__(mm_related_params)
        if getattr(self.visual.config, "deepstack_visual_indexes", None):
            raise ValueError(
                "Qwen4 adapter requires the released non-DeepStack vision tower"
            )

    preprocess_input = staticmethod(preprocess_qwen4)

    def get_preprocess_params(self):
        return {"processor": self.mm_processor, "video_backend": self.video_backend}

    def token_length(self, mm_input, vit_config):
        return qwen4_token_length(mm_input, vit_config, **self.get_preprocess_params())

    def estimate_work(self, data, mm_type=None):
        grid = data.grid
        if grid.device.type != "cpu" or tuple(grid.shape) != (1, 3):
            raise ValueError("Qwen4 requires one CPU grid per media item")
        t, h, w = grid[0].tolist()
        merge = self.visual.spatial_merge_size
        patches = t * h * w
        if (
            min(t, h, w) <= 0
            or h % merge
            or w % merge
            or data.pixels.shape[0] != patches
        ):
            raise ValueError(f"invalid Qwen4 vision grid: {(t, h, w)}")
        if data.timestamps is None:
            text_tokens = 2
        else:
            text_tokens = sum(
                2
                + len(
                    self.mm_processor.tokenizer.encode(
                        f"<{timestamp:.1f} seconds>", add_special_tokens=False
                    )
                )
                for timestamp in data.timestamps
            )
        config = self.visual.config
        return MMWorkEstimate(
            input_patches=patches,
            output_tokens=patches // merge**2 + text_tokens,
            estimated_workspace_bytes=patches
            * torch.empty((), dtype=self._data_type).element_size()
            * (8 * config.hidden_size + 2 * config.intermediate_size)
            + getattr(data.pixels, "workspace_bytes", 0),
            max_attention_segment=h * w,
            attention_work=t * (h * w) ** 2,
        )

    def prepare_embedding_inputs(self, data_list, mm_types):
        pairs = [(data.pixels, data.grid) for data in data_list]
        prepared = super().prepare_embedding_inputs(pairs, mm_types)
        return [
            replace(data, pixels=pixels) if pixels is not data.pixels else data
            for data, (pixels, _) in zip(data_list, prepared)
        ]

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return self.batched_embedding([data], [None], **kwargs)[0]

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        if not data_list:
            return []
        embeddings = self._batched_vision_forward(
            [(data.pixels, data.grid) for data in data_list], **kwargs
        )
        merge = self.visual.spatial_merge_size
        features = embeddings.split(
            [int(data.grid.prod()) // merge**2 for data in data_list]
        )
        return [
            assemble_visual_embedding(
                data,
                feature,
                self.get_position_ids(data.grid)[0],
                self.mm_processor,
                self.text_embedding,
            )
            for data, feature in zip(data_list, features)
        ]


class Qwen4ExpMixin(Qwen3_5MoeMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen4ExpImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )

    def load_mm_weight(self, ctype, device):
        super().load_mm_weight(ctype, device)
        self.mm_part.text_embedding = self.mm_part.word_embedding_weight

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        return Qwen4ExpImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen4_exp"], Qwen4ExpMixin)
