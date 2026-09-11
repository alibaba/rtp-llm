import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLImageProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import (
    GpuVideoInput,
    prepare_gpu_video,
    preprocess_video_cuda,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
    Qwen3_VLImageEmbedding,
    Qwen3_VLMixin,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.database import CkptDatabase


class Qwen3_5MoeImageEmbedding(Qwen3_VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        self.video_backend = os.environ.get("QWEN35_VIDEO_BACKEND", "cpu")
        if self.video_backend not in ("cpu", "nvdec"):
            raise ValueError("QWEN35_VIDEO_BACKEND must be cpu or nvdec")
        logging.info("Qwen3.5 video preprocessing backend: %s", self.video_backend)
        self.mm_processor = AutoProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        self.mm_processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        config_hf = Qwen3_5MoeVisionConfig.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        # The vision tower selects its packed backend from the actual tensor device.
        config_hf._attn_implementation = "sdpa"
        config_hf.vit_attention_backend = os.environ.get(
            "QWEN35_VIT_ATTN_BACKEND", "auto"
        )
        self.visual = Qwen3_5MoeVisionModel._from_config(config_hf)

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "video_backend": self.video_backend,
            "factor": self.visual.spatial_merge_size * self.visual.patch_size,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs, vit_config, processor, factor=32, video_backend="cpu"
    ):
        if (
            video_backend == "nvdec"
            and len(mm_inputs) == 1
            and mm_inputs[0].mm_type == MMUrlType.VIDEO
        ):
            item = mm_inputs[0]
            source = get_bytes_io_from_url(
                item.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_video_max_file_size_kb,
            )
            return prepare_gpu_video(
                source.getvalue(),
                item.mm_preprocess_config,
                processor.video_processor,
                factor,
            )
        return Qwen3_VLImageEmbedding.preprocess_input(
            mm_inputs, vit_config, processor, factor
        )

    def estimate_work(self, data, mm_type=None) -> MMWorkEstimate:
        grid = data[1]
        if grid.device.type != "cpu":
            raise ValueError("Qwen3.5 scheduling metadata must be on CPU")
        rows = grid.tolist()
        merge = self.visual.spatial_merge_size
        if not rows or any(
            t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge for t, h, w in rows
        ):
            raise ValueError(f"invalid Qwen3.5 vision grid: {rows}")
        patches = sum(t * h * w for t, h, w in rows)
        if data[0].shape[0] != patches:
            raise ValueError("pixel_values length does not match grid_thw")
        config = self.visual.config
        workspace_per_patch = torch.empty((), dtype=self._data_type).element_size() * (
            8 * config.hidden_size + 2 * config.intermediate_size
        )
        return MMWorkEstimate(
            input_patches=patches,
            output_tokens=patches // merge**2,
            estimated_workspace_bytes=(
                patches * workspace_per_patch
                + (data[0].workspace_bytes if isinstance(data[0], GpuVideoInput) else 0)
            ),
            max_attention_segment=max(h * w for _, h, w in rows),
            attention_work=sum(t * (h * w) ** 2 for t, h, w in rows),
        )

    def get_batch_work_budget(self, max_batch_media: int):
        if max_batch_media >= 1 << 30:
            return None
        # One reference item is bounded by the model's configured image/video
        # pixel policy. Variable media count alone cannot bound GPU work.
        image = self.mm_processor.image_processor
        video = self.mm_processor.video_processor
        image_pixels = getattr(image, "max_pixels", 0) or image.size.get(
            "longest_edge", 0
        )
        video_pixels = video.size["longest_edge"] // video.temporal_patch_size
        patches = max(image_pixels, video_pixels) // self.visual.patch_size**2
        config = self.visual.config
        workspace_per_patch = torch.empty((), dtype=self._data_type).element_size() * (
            8 * config.hidden_size + 2 * config.intermediate_size
        )
        return MMWorkEstimate(
            input_patches=patches,
            output_tokens=patches // self.visual.spatial_merge_size**2,
            estimated_workspace_bytes=patches * workspace_per_patch,
            max_attention_segment=patches,
            attention_work=patches**2,
        ).scaled(max_batch_media)

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return self.batched_embedding(
            [data], [kwargs.pop("mm_type", MMUrlType.IMAGE)], **kwargs
        )[0]

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        if len(data_list) != len(mm_types):
            raise ValueError("data and media type counts differ")
        if not data_list:
            return []
        if not all(
            t in (MMUrlType.DEFAULT, MMUrlType.IMAGE, MMUrlType.VIDEO) for t in mm_types
        ):
            raise ValueError("Qwen3.5 vision supports only image and video inputs")
        estimates = [
            self.estimate_work(data, kind) for data, kind in zip(data_list, mm_types)
        ]
        pixels = [
            (
                preprocess_video_cuda(
                    data[0], self.mm_processor.video_processor, self._device
                )
                if isinstance(data[0], GpuVideoInput)
                else data[0]
            ).to(device=self._device, dtype=self._data_type)
            for data in data_list
        ]
        pixel_values = pixels[0] if len(pixels) == 1 else torch.cat(pixels, dim=0)
        # Keep shape bookkeeping on CPU instead of copying it back from CUDA
        # inside every vision layer.
        grid_thw = torch.cat([data[1] for data in data_list], dim=0)
        embeds = self.visual(
            pixel_values, grid_thw=grid_thw, return_dict=True, **kwargs
        ).pooler_output
        per_item_embeds = embeds.split(
            [estimate.output_tokens for estimate in estimates]
        )
        per_grid_positions = self.get_position_ids(grid_thw, device=self._device)
        results, grid_offset = [], 0
        for data, embedding in zip(data_list, per_item_embeds):
            count = data[1].shape[0]
            positions = per_grid_positions[grid_offset : grid_offset + count]
            results.append(
                (embedding, positions[0] if count == 1 else torch.cat(positions))
            )
            grid_offset += count
        return results

    def get_position_ids(
        self, grid_thw: torch.Tensor = None, device=None
    ) -> List[torch.Tensor]:
        spatial_merge_size = self.visual.spatial_merge_size
        device = grid_thw.device if device is None else device
        grid_thw = grid_thw.cpu()
        dtype = torch.int32

        t_all = grid_thw[:, 0].to(dtype)
        h_all = (grid_thw[:, 1] // spatial_merge_size).to(dtype)
        w_all = (grid_thw[:, 2] // spatial_merge_size).to(dtype)

        pos_list = []
        for t, h, w in zip(t_all.tolist(), h_all.tolist(), w_all.tolist()):
            if t == 0 or h == 0 or w == 0:
                pos_list.append(torch.empty((0, 3), device=device, dtype=dtype))
                continue

            t_grid = (
                torch.arange(t, device=device, dtype=dtype)
                .view(t, 1, 1)
                .expand(-1, h, w)
            )
            h_grid = (
                torch.arange(h, device=device, dtype=dtype)
                .view(1, h, 1)
                .expand(t, -1, w)
            )
            w_grid = (
                torch.arange(w, device=device, dtype=dtype)
                .view(1, 1, w)
                .expand(t, h, -1)
            )

            pos = torch.stack(
                (t_grid.reshape(-1), h_grid.reshape(-1), w_grid.reshape(-1)), dim=1
            )
            pos_list.append(pos)

        return pos_list


class Qwen3_5MoeVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."

    def detect_ckpt_prefix(self, tensor_names: List[str]):
        suffix = "blocks.0.norm1.weight"
        suffix_with_visual = "visual." + suffix
        for name in tensor_names:
            if name.endswith(suffix_with_visual):
                # Extract the prefix ending at the visual module closest to the anchor.
                self._ckpt_prefix = name[: -len(suffix)]
                break
        else:
            raise ValueError(
                f"Qwen3_5MoeVitWeight: cannot determine visual prefix, no key ending "
                f"with {suffix_with_visual!r} in {len(tensor_names)} ckpt keys"
            )


class Qwen3_5MoeMixin(Qwen3_VLMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen3_5MoeImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )

    def _prepare_vit_weights(self, database: CkptDatabase) -> None:
        vit_weights = self.mm_related_params.vit_weights
        if isinstance(vit_weights, Qwen3_5MoeVitWeight):
            vit_weights.detect_ckpt_prefix(database.get_pretrain_tensor_names())

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        # Inherits from Qwen3_VLMixin but uses Qwen3_5MoeImageEmbedding for the ViT;
        # without this override, eval_mm_model_size would instantiate the wrong class
        # and underreport / mis-shape the ViT parameter count.
        return Qwen3_5MoeImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen35_moe"], Qwen3_5MoeMixin)
register_multimodal_mixin(["qwen35_dense"], Qwen3_5MoeMixin)
register_multimodal_mixin(["qwen35_dense_mtp"], Qwen3_5MoeMixin)
register_multimodal_mixin(["qwen35_moe_mtp"], Qwen3_5MoeMixin)
