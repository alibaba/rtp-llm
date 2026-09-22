import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLImageProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import (
    GpuVideoInput,
    PreparedGpuVideo,
    VideoDecodePool,
    prepare_gpu_video,
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
from rtp_llm.multimodal.qwen3_vl_video import resolve_video_size, video_timestamp_tokens
from rtp_llm.multimodal.vit_metrics import (
    record_vit_preprocess_value,
    vit_preprocess_timer,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.database import CkptDatabase


class Qwen3_5MoeImageEmbedding(Qwen3_VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        self._video_pool_lock = threading.Lock()
        self.video_decode_workers = int(
            os.environ.get("QWEN35_VIDEO_DECODE_WORKERS", "16")
        )
        if self.video_decode_workers < 1:
            raise ValueError("QWEN35_VIDEO_DECODE_WORKERS must be positive")
        self.video_backend = os.environ.get("QWEN35_VIDEO_BACKEND", "nvdec")
        if self.video_backend not in ("cpu", "nvdec"):
            raise ValueError("QWEN35_VIDEO_BACKEND must be cpu or nvdec")
        logging.info(
            "Qwen3.5 video preprocessing backend: %s, decode workers: %d",
            self.video_backend,
            self.video_decode_workers,
        )
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
        self.word_embedding_weight = None
        tokenizer = self.mm_processor.tokenizer
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        if self.vision_start_token_id is None or self.vision_end_token_id is None:
            raise ValueError("video tokenizer lacks vision delimiters")

        from .vision_graph import VisionGraphCache

        self._vision_graph_cache = VisionGraphCache(self.visual)

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
        mm_inputs, vit_config, processor, factor=32, video_backend="nvdec"
    ):
        if (
            video_backend == "nvdec"
            and len(mm_inputs) == 1
            and mm_inputs[0].mm_type == MMUrlType.VIDEO
        ):
            item = mm_inputs[0]
            with vit_preprocess_timer(
                GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC,
                {"model": "qwen35", "mm_type": "video"},
            ):
                source = get_bytes_io_from_url(
                    item.url,
                    vit_config.download_headers,
                    max_file_size_kb=vit_config.mm_video_max_file_size_kb,
                )
            data, grid = prepare_gpu_video(
                source.getvalue(),
                item.mm_preprocess_config,
                processor.video_processor,
                factor,
                size=resolve_video_size(
                    processor.video_processor,
                    vit_config.mm_video_total_min_pixels,
                    vit_config.mm_video_total_max_pixels,
                ),
            )
            timestamps = video_timestamp_tokens(
                grid, data.frame_indices, data.source_fps, processor
            )
            return data, grid, timestamps
        data = Qwen3_VLImageEmbedding.preprocess_input(
            mm_inputs, vit_config, processor, factor, return_video_metadata=True
        )
        if mm_inputs[0].mm_type == MMUrlType.VIDEO:
            pixels, grid, metadata = data
            record_vit_preprocess_value(
                GaugeMetrics.VIT_VIDEO_FRAME_COUNT_METRIC,
                len(metadata.frames_indices),
                {"model": "qwen35", "mm_type": "video", "backend": "cpu"},
            )
            return (
                pixels,
                grid,
                video_timestamp_tokens(
                    grid, metadata.frames_indices, metadata.fps, processor
                ),
            )
        return data

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
        text_tokens = 0
        if mm_type == MMUrlType.VIDEO:
            if len(data) != 3 or len(rows) != 1 or len(data[2]) != rows[0][0]:
                raise ValueError("video timestamps and grid do not match")
            text_tokens = sum(len(ids) + 2 for ids in data[2])
        config = self.visual.config
        workspace_per_patch = torch.empty((), dtype=self._data_type).element_size() * (
            8 * config.hidden_size + 2 * config.intermediate_size
        )
        return MMWorkEstimate(
            input_patches=patches,
            output_tokens=patches // merge**2 + text_tokens,
            estimated_workspace_bytes=(
                patches * workspace_per_patch
                + text_tokens
                * (
                    config.out_hidden_size
                    * torch.empty((), dtype=self._data_type).element_size()
                    + 12
                )
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

    def _get_video_pool(self):
        with self._video_pool_lock:
            if not hasattr(self, "_video_decode_pool"):
                self._video_decode_pool = VideoDecodePool(self.video_decode_workers)
            return self._video_decode_pool

    def prepare_embedding_inputs(self, data_list, mm_types):
        """Run in admitted request threads, before the single ViT executor."""
        videos = [data[0] for data in data_list if isinstance(data[0], GpuVideoInput)]
        if not videos:
            return data_list
        prepared = []
        with self._get_video_pool().preprocess(
            videos, self.mm_processor.video_processor, self._device, self._data_type
        ) as results:
            for data in data_list:
                if isinstance(data[0], GpuVideoInput):
                    prepared.append((next(results), *data[1:]))
                else:
                    prepared.append(data)
        return prepared

    def _preprocess_batch(self, data_list):
        # Also support direct model calls that do not pass through MMScheduler.
        data_list = self.prepare_embedding_inputs(data_list, None)
        pixels = []
        for data in data_list:
            value = data[0]
            if isinstance(value, PreparedGpuVideo):
                value = value.consume(torch.cuda.current_stream(self._device))
            pixels.append(value.to(device=self._device, dtype=self._data_type))
        return pixels

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
        pixels = self._preprocess_batch(data_list)
        pixel_values = pixels[0] if len(pixels) == 1 else torch.cat(pixels, dim=0)
        # Keep shape bookkeeping on CPU instead of copying it back from CUDA
        # inside every vision layer.
        grid_thw = torch.cat([data[1] for data in data_list], dim=0)
        # Lazy construction also supports weight-loading and test-created modules.
        if not hasattr(self, "_vision_graph_cache"):
            from .vision_graph import VisionGraphCache

            self._vision_graph_cache = VisionGraphCache(self.visual)
        embeds = self._vision_graph_cache.run(
            pixel_values, grid_thw, return_dict=True, **kwargs
        )
        per_item_embeds = embeds.split(
            [
                estimate.input_patches // self.visual.spatial_merge_size**2
                for estimate in estimates
            ]
        )
        # One CPU lookup and one H2D copy for every video's timestamp/tag tokens
        # in this scheduler batch. The full vocabulary stays on CPU.
        text_ids = [
            token
            for data, kind in zip(data_list, mm_types)
            if kind == MMUrlType.VIDEO
            for ids in data[2]
            for token in [*ids, self.vision_start_token_id, self.vision_end_token_id]
        ]
        text_embeddings = None
        if text_ids:
            if self.word_embedding_weight is None:
                raise RuntimeError(
                    "Qwen3.5 video word embedding weights are not loaded"
                )
            text_embeddings = self.word_embedding_weight[
                torch.tensor(text_ids, dtype=torch.long)
            ].to(device=self._device, dtype=self._data_type)
        text_offset = 0
        results = []
        for data, embedding, mm_type in zip(data_list, per_item_embeds, mm_types):
            if mm_type == MMUrlType.VIDEO:
                embedding, position_ids, text_offset = self._assemble_video(
                    embedding, data[1], data[2], text_embeddings, text_offset
                )
            else:
                positions = self.get_position_ids(data[1], device=self._device)
                position_ids = (
                    positions[0] if len(positions) == 1 else torch.cat(positions)
                )
            results.append((embedding, position_ids))
        return results

    def load_word_embedding(self, database: CkptDatabase):
        # Load with the model database so both safetensors and supported legacy
        # checkpoints work. Keep the vocabulary off the ViT GPU and out of its
        # parameter-size probe.
        candidates = (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
        names = set(database.get_pretrain_tensor_names())
        key = next((key for key in candidates if key in names), None)
        if key is None:
            raise ValueError("Qwen3.5 checkpoint has no language word embedding")
        weights = database.load_tensor(key, data_type=None)
        if len(weights) != 1 or weights[0].ndim != 2:
            raise ValueError(
                "Qwen3.5 word embedding must be one full vocabulary matrix"
            )
        weight = weights[0]
        if (
            weight.shape[1] != self.visual.config.out_hidden_size
            or weight.shape[0]
            <= max(self.vision_start_token_id, self.vision_end_token_id)
            or not weight.is_floating_point()
        ):
            raise ValueError(
                "Qwen3.5 word embedding shape/dtype does not match the vision output"
            )
        self.word_embedding_weight = weight.detach().cpu()
        logging.info(
            "Loaded Qwen3.5 video word embedding %s on CPU: %s",
            key,
            tuple(weight.shape),
        )

    def _assemble_video(self, features, grid, timestamps, text_embeddings, text_offset):
        """Return a whole video span and its relative 3D MRoPE positions.

        The prompt's outer vision tags remain ordinary text. Inside the span,
        each timestamp and frame tag uses a word embedding. Each frame's visual
        tokens reset temporal coordinates to zero; following text advances past
        the largest spatial coordinate, exactly as separate frame spans do.
        """
        t, h, w = grid[0].tolist()
        h //= self.visual.spatial_merge_size
        w //= self.visual.spatial_merge_size
        frame_size = h * w
        if features.shape[0] != t * frame_size:
            raise ValueError("video features and grid do not match")
        frame_positions = torch.stack(
            (
                torch.zeros(frame_size, dtype=torch.int32),
                torch.arange(h, dtype=torch.int32).repeat_interleave(w),
                torch.arange(w, dtype=torch.int32).repeat(h),
            ),
            dim=1,
        )
        chunks, positions = [], []
        base = 0
        for frame, ids in enumerate(timestamps):
            prefix_length = len(ids) + 1  # timestamp and vision_start
            chunks.extend(
                (
                    text_embeddings[text_offset : text_offset + prefix_length],
                    features[frame * frame_size : (frame + 1) * frame_size],
                    text_embeddings[
                        text_offset + prefix_length : text_offset + prefix_length + 1
                    ],
                )
            )
            text_offset += prefix_length + 1
            positions.append(
                torch.arange(base, base + prefix_length, dtype=torch.int32)[
                    :, None
                ].expand(-1, 3)
            )
            base += prefix_length
            positions.append(frame_positions + base)
            base += max(h, w)
            positions.append(torch.full((1, 3), base, dtype=torch.int32))
            base += 1
        return (
            torch.cat(chunks, dim=0),
            torch.cat(positions, dim=0).to(device=features.device),
            text_offset,
        )

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

    def load_mm_weight(self, ctype: str, device: str):
        from rtp_llm.utils.util import to_torch_dtype

        if not self.weights:
            raise RuntimeError(
                f"No multimodal weights loaded from {self.ckpt_path!r}; "
                "check checkpoint path and mixin configuration."
            )
        # Preserve the vLLM Parameter subclasses and invoke their QKV/row/column
        # loaders. The generic RTP loader assigns param.data directly.
        visual = self.mm_part.visual
        visual.to(device=device, dtype=to_torch_dtype(ctype))
        visual.load_weights(self.weights.items())

    def _prepare_vit_weights(self, database: CkptDatabase) -> None:
        vit_weights = self.mm_related_params.vit_weights
        if isinstance(vit_weights, Qwen3_5MoeVitWeight):
            vit_weights.detect_ckpt_prefix(database.get_pretrain_tensor_names())
        self.mm_part.load_word_embedding(database)

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
