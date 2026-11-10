"""Thin Qwen4 adapter over the existing Qwen3.5 vision tower."""

import itertools
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    Qwen3_5MoeMixin,
    Qwen3_5MoeVitWeight,
    default_attn_impl,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_processor import preprocess_qwen4


class Qwen4ExpImageEmbedding(Qwen3_5MoeImageEmbedding):
    def __init__(self, mm_related_params):
        checkpoint = mm_related_params.config["ckpt_path"]
        config = json.loads((Path(checkpoint) / "config.json").read_text())[
            "vision_config"
        ]
        if config.get("deepstack_visual_indexes"):
            raise ValueError(
                "Qwen4 adapter requires the released non-DeepStack vision tower"
            )
        self.mm_processor = AutoProcessor.from_pretrained(checkpoint)
        config = Qwen3_5MoeVisionConfig(**config)
        config._attn_implementation = default_attn_impl
        self.visual = Qwen3_5MoeVisionModel._from_config(config)

    preprocess_input = staticmethod(preprocess_qwen4)

    def get_preprocess_params(self):
        return {"processor": self.mm_processor}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return self.batched_embedding([data], [None], **kwargs)[0]

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        pixels = torch.cat([data.pixels for data in data_list])
        grids = torch.cat([data.grid for data in data_list])
        if grids.device.type != "cpu":
            raise ValueError("Qwen4 grid metadata must be produced on CPU")
        # Keep model-produced geometry on the host. The shared tower can then
        # reuse explicit lengths without reading its device cu_seqlens per layer.
        lengths = tuple(h * w for t, h, w in grids.tolist() for _ in range(t))
        pinned = self._device.type == "cuda"
        cu = torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=torch.int32, pin_memory=pinned
        ).to(device=self._device, non_blocking=True)
        if pinned:
            pixels = pixels.pin_memory()
        pixels = pixels.to(
            device=self._device, dtype=self._data_type, non_blocking=True
        )
        output = self.visual(
            pixels,
            grid_thw=grids,
            cu_seqlens=cu,
            sequence_lengths=lengths,
            async_metadata=True,
            return_dict=True,
            **kwargs
        ).pooler_output
        sizes = [
            sum(length for _, length in data.token_plan.spans) for data in data_list
        ]
        results = []
        for features, data in zip(output.split(sizes), data_list):
            lengths = [length for _, length in data.token_plan.spans]
            segments = list(features.split(lengths))
            data.token_plan.validate(segments)
            results.append(
                (segments, data.positions, data.semantic_ids, data.token_plan)
            )
        return results


class Qwen4ExpMixin(Qwen3_5MoeMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen4ExpImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        return Qwen4ExpImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen4_exp"], Qwen4ExpMixin)
