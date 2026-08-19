"""
MiniMax-M3 VL multimodal mixin.

Handles both image and video inputs.  For images the HF processor produces
``]<]start of image[>[`` + N × ``]<]image[>[`` + ``]<]end of image[>[``;
for video each temporal group gets a timestamp prefix
``]<]X.X seconds[>[`` + the same bracket/token pattern.

This mixin reproduces the same token-count contract by:

1. Running the ViT + projector to get patch features (N rows).
2. Looking up the LLM word-embedding vectors for the bracket / timestamp
   tokens and concatenating them around the ViT features.
3. Returning one flat ``(total_tokens, hidden_dim)`` tensor so the C++
   ``expandTokenIds`` single-token mode replaces the 1 placeholder token
   with the correct number of embedding rows — no C++ changes needed.

The LLM word-embedding table is loaded once from the same checkpoint at
init time (kept on CPU, ~2.3 GB); fixed bracket embeddings are cached on
GPU.
"""

import json
import logging
import math
import os
import threading
from collections import OrderedDict
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters
from rtp_llm.utils.model_weight import CkptWeightInfo, concat_0, identity, sp_id

from .image_processor import (
    IMAGE_MAX_TOTAL_PIXELS,
    MIN_SHORT_SIDE_PIXEL,
    VIDEO_MAX_TOTAL_PIXELS,
    MiniMaxM3VLImageProcessor,
    compute_sampled_frame_indices,
    get_hw_multiple_of,
    smart_resize,
)
from .minimax_m3_vl_vit import (  # noqa: F401
    MiniMaxM3VLVisionTower,
    VisionConfig,
    _select_attention_backend,
    get_fused_qkv_checkpoint_names,
)

logger = logging.getLogger(__name__)

_VIDEO_DECODE_CHUNK_FRAMES = 16


class _MiniMaxM3VLPreprocessBuffers(nn.Module):
    """FP32 normalization constants that follow the active ViT device."""

    def __init__(self, image_mean, image_std):
        super().__init__()
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )


class _MiniMaxM3VLVisionGraphEntry:
    def __init__(self, graph, static_input, static_output, attention_context):
        self.graph = graph
        self.static_input = static_input
        self.static_output = static_output
        self.attention_context = attention_context
        self.ready_event = torch.cuda.Event()
        self.has_pending_output = False


class _MiniMaxM3VLVisionGraphCache:
    """Bounded exact-shape CUDA Graph cache for the M3VL vision tower.

    Packed attention isolation depends on the complete grid signature, so graph
    entries are never reused across different grids. A signature is captured on
    its second occurrence; one-off or unsupported workloads stay eager.
    """

    _GRAPH_BACKENDS = frozenset(("fa4", "flash_attn", "flashinfer"))

    def __init__(
        self,
        visual: nn.Module,
        max_entries: int = 4,
        capture_after: int = 2,
        max_graph_patches: int = 4096,
    ):
        self._visual = visual
        self._max_entries = max_entries
        self._capture_after = capture_after
        self._max_graph_patches = max_graph_patches
        self._enabled = True
        self._entries = OrderedDict()
        self._seen = {}
        self._disabled = set()
        self._lock = threading.Lock()
        self._stats = {
            "hit": 0,
            "miss": 0,
            "capture": 0,
            "fallback": 0,
        }

    @staticmethod
    def _signature(pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        grid = tuple(tuple(int(value) for value in row) for row in grid_thw.tolist())
        return (
            pixel_values.device.type,
            pixel_values.device.index,
            pixel_values.dtype,
            tuple(pixel_values.shape),
            grid,
        )

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def set_enabled(self, enabled: bool):
        with self._lock:
            self._enabled = enabled

    def _fallback(self, pixel_values, grid_thw):
        self._stats["fallback"] += 1
        kmonitor.report(AccMetrics.VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC, 1)
        return self._visual(pixel_values, grid_thw)

    def _capture(self, pixel_values, grid_thw):
        static_input = pixel_values.detach().clone()
        vision_model = getattr(
            getattr(self._visual, "vision_tower", None),
            "vision_model",
            self._visual,
        )
        attention_context = vision_model.prepare_cuda_graph_attention_context(
            grid_thw,
            pixel_values.device,
            pixel_values.dtype,
        )
        current_stream = torch.cuda.current_stream(pixel_values.device)
        capture_stream = torch.cuda.Stream(device=pixel_values.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            self._visual(
                static_input,
                grid_thw,
                attention_context=attention_context,
            )
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(pixel_values.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_output = self._visual(
                static_input,
                grid_thw,
                attention_context=attention_context,
            )
        return _MiniMaxM3VLVisionGraphEntry(
            graph,
            static_input,
            static_output,
            attention_context,
        )

    @staticmethod
    def _replay(entry, pixel_values):
        stream = torch.cuda.current_stream(pixel_values.device)
        if entry.has_pending_output:
            stream.wait_event(entry.ready_event)
        entry.static_input.copy_(pixel_values)
        entry.graph.replay()
        output = entry.static_output.clone()
        entry.ready_event.record(stream)
        entry.has_pending_output = True
        return output

    def run(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        if not self._enabled:
            return self._visual(pixel_values, grid_thw)
        # Packed batches have data-dependent segment counts and graph I/O copies
        # outweighed launch savings in profiling. Keep them eager until Stage 6
        # has a padding scheme that preserves segment-level attention isolation.
        if grid_thw.ndim != 2 or grid_thw.shape[0] != 1:
            return self._visual(pixel_values, grid_thw)
        if (
            not pixel_values.is_cuda
            or pixel_values.shape[0] > self._max_graph_patches
            or torch.cuda.is_current_stream_capturing()
            or _select_attention_backend(pixel_values) not in self._GRAPH_BACKENDS
        ):
            with self._lock:
                return self._fallback(pixel_values, grid_thw)

        signature = self._signature(pixel_values, grid_thw)
        with self._lock:
            entry = self._entries.get(signature)
            if entry is not None:
                self._entries.move_to_end(signature)
                self._stats["hit"] += 1
                kmonitor.report(AccMetrics.VIT_CUDA_GRAPH_HIT_QPS_METRIC, 1)
                kmonitor.report(GaugeMetrics.VIT_CUDA_GRAPH_PADDING_RATIO_METRIC, 0)
                return self._replay(entry, pixel_values)

            self._stats["miss"] += 1
            kmonitor.report(AccMetrics.VIT_CUDA_GRAPH_MISS_QPS_METRIC, 1)
            if signature in self._disabled:
                return self._visual(pixel_values, grid_thw)

            seen = self._seen.get(signature, 0) + 1
            self._seen[signature] = seen
            if seen < self._capture_after:
                return self._visual(pixel_values, grid_thw)

            try:
                entry = self._capture(pixel_values, grid_thw)
            except (RuntimeError, AssertionError, ValueError) as error:
                self._disabled.add(signature)
                self._stats["fallback"] += 1
                kmonitor.report(AccMetrics.VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC, 1)
                logger.warning(
                    "M3VL vision CUDA Graph capture failed; using eager: %s",
                    error,
                )
                return self._visual(pixel_values, grid_thw)

            self._entries[signature] = entry
            self._seen.pop(signature, None)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            # Capture records the work but some graph-safe kernels do not leave
            # their captured output buffer in a consumable state. Replay once so
            # the request that triggered capture observes the same result as hits.
            self._stats["capture"] += 1
            kmonitor.report(AccMetrics.VIT_CUDA_GRAPH_CAPTURE_QPS_METRIC, 1)
            kmonitor.report(GaugeMetrics.VIT_CUDA_GRAPH_PADDING_RATIO_METRIC, 0)
            logger.info(
                "captured M3VL vision CUDA Graph shape=%s grids=%s cache_size=%d",
                tuple(pixel_values.shape),
                signature[-1],
                len(self._entries),
            )
            return self._replay(entry, pixel_values)


class MiniMaxM3VLImageEmbedding(MultiModalEmbeddingInterface):
    """
    Wraps the MiniMax-M3 VL vision stack.  Produces a single flat embedding
    tensor that includes bracket / timestamp word-embeddings around the ViT
    patch features so the C++ single-token expansion gets the right total
    token count.
    """

    # Token IDs for fixed bracket tokens (single-token special tokens).
    START_IMAGE_TOKEN_ID = 200029  # ]<]start of image[>[
    END_IMAGE_TOKEN_ID = 200030  # ]<]end of image[>[
    MAX_IMAGES_PER_REQUEST = 200
    MAX_VIDEOS_PER_REQUEST = 20
    MIN_VIDEO_FPS = 0.2
    MAX_VIDEO_FPS = 5.0

    def validate_inputs(self, mm_inputs: List[MultimodalInput]) -> None:
        image_count = sum(
            mm_input.mm_type in (MMUrlType.DEFAULT, MMUrlType.IMAGE)
            for mm_input in mm_inputs
        )
        video_count = sum(mm_input.mm_type == MMUrlType.VIDEO for mm_input in mm_inputs)
        if image_count > self.MAX_IMAGES_PER_REQUEST:
            raise_mm(
                MMErr.IMAGE_REQ.format(
                    f"at most {self.MAX_IMAGES_PER_REQUEST} images are allowed, "
                    f"got {image_count}"
                )
            )
        if video_count > self.MAX_VIDEOS_PER_REQUEST:
            raise_mm(
                MMErr.VIDEO_REQ.format(
                    f"at most {self.MAX_VIDEOS_PER_REQUEST} videos are allowed, "
                    f"got {video_count}"
                )
            )
        for mm_input in mm_inputs:
            if mm_input.mm_type != MMUrlType.VIDEO:
                continue
            fps = float(getattr(mm_input.mm_preprocess_config, "fps", -1))
            if fps > 0 and (
                not math.isfinite(fps)
                or fps < self.MIN_VIDEO_FPS
                or fps > self.MAX_VIDEO_FPS
            ):
                raise_mm(
                    MMErr.VIDEO_REQ.format(
                        f"fps must be in [{self.MIN_VIDEO_FPS}, "
                        f"{self.MAX_VIDEO_FPS}], got {fps}"
                    )
                )

    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]

        self.hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        self.mm_processor = MiniMaxM3VLImageProcessor.from_pretrained(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True
        )

        self.image_token_index = getattr(self.hf_config, "image_token_index", 200025)
        self.video_token_index = getattr(self.hf_config, "video_token_index", 200026)

        self.visual = MiniMaxM3VLVisionTower(self.hf_config).to(torch.bfloat16)
        self._preprocess_buffers = _MiniMaxM3VLPreprocessBuffers(
            self.mm_processor.image_mean,
            self.mm_processor.image_std,
        )
        self._vision_graph_cache = _MiniMaxM3VLVisionGraphCache(self.visual)

        # --- LLM word embedding (CPU, ~2.3 GB) ---
        self.word_embedding_weight = self._load_word_embedding(ckpt_path)

        # Pre-extract fixed bracket embeddings (will be moved to GPU lazily).
        self._start_emb = self.word_embedding_weight[self.START_IMAGE_TOKEN_ID].clone()
        self._end_emb = self.word_embedding_weight[self.END_IMAGE_TOKEN_ID].clone()
        self._bracket_embs_on_device = False

        # --- Video sampling defaults (from HF video processor config) ---
        self.video_fps = float(getattr(self.hf_config, "fps", 1.0))
        self.video_max_frames = int(getattr(self.hf_config, "max_frames", 768))
        self.video_min_frames = int(getattr(self.hf_config, "min_frames", 4))
        self.temporal_patch_size = self.mm_processor.temporal_patch_size
        self.merge_size = self.mm_processor.merge_size

    @staticmethod
    def _load_word_embedding(ckpt_path: str) -> torch.Tensor:
        from safetensors import safe_open

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        emb_key = "language_model.model.embed_tokens.weight"
        with open(index_path) as f:
            index = json.load(f)
        shard = index["weight_map"][emb_key]
        shard_path = os.path.join(ckpt_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            return sf.get_tensor(emb_key)

    def _ensure_bracket_embs_on_device(self):
        if not self._bracket_embs_on_device:
            device = self._device
            dtype = self._data_type
            self._start_emb = self._start_emb.to(device=device, dtype=dtype)
            self._end_emb = self._end_emb.to(device=device, dtype=dtype)
            self._bracket_embs_on_device = True

    def _ensure_preprocess_buffers_on_device(self):
        if self._preprocess_buffers.image_mean.device != self._device:
            self._preprocess_buffers.to(device=self._device)
        return (
            self._preprocess_buffers.image_mean,
            self._preprocess_buffers.image_std,
        )

    def vision_graph_stats(self):
        return self._vision_graph_cache.stats()

    def set_vision_cuda_graph_enabled(self, enabled: bool):
        self._vision_graph_cache.set_enabled(enabled)

    def _run_visual(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        return self._vision_graph_cache.run(pixel_values, grid_thw)

    @property
    def _data_type(self) -> torch.dtype:
        return self.visual.dtype

    @property
    def _device(self):
        return next(self.visual.parameters()).device

    def _workspace_bytes_per_patch(self) -> int:
        """Conservative live-activation estimate for one ViT input patch."""
        config = self.visual.vision_config
        dtype_bytes = torch.empty((), dtype=self._data_type).element_size()
        # QKV/RoPE/attention temporaries plus the two MLP projections dominate
        # peak inference activation memory. For the released M3VL config this is
        # 40 KiB/patch, slightly above the 35-37 KiB measured in Stage 1.5.
        activation_elements = 8 * config.hidden_size + 2 * config.intermediate_size
        return dtype_bytes * activation_elements

    def estimate_work(
        self, data: Any, mm_type: Optional[MMUrlType] = None
    ) -> MMWorkEstimate:
        """Compute exact M3VL work from the CPU preprocess result.

        This deliberately mirrors ``_gpu_fold`` without running resize, fold,
        patch embedding, or any other GPU operation.
        """
        raw, target_hw, timestamp_token_ids = data
        target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
        patch_size = int(self.mm_processor.patch_size)
        merge_size = int(self.merge_size)
        temporal_patch_size = int(self.temporal_patch_size)

        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"invalid M3VL target size {target_hw}")
        if target_h % patch_size != 0 or target_w % patch_size != 0:
            raise ValueError(
                f"M3VL target size {target_hw} is not patch-aligned " f"to {patch_size}"
            )

        is_video = timestamp_token_ids is not None
        frame_count = int(raw.shape[0]) if is_video else 1
        if frame_count <= 0:
            raise ValueError("M3VL preprocess result has no frames")

        grid_t = (frame_count + temporal_patch_size - 1) // temporal_patch_size
        grid_h = target_h // patch_size
        grid_w = target_w // patch_size
        if grid_h % merge_size != 0 or grid_w % merge_size != 0:
            raise ValueError(
                f"M3VL patch grid {(grid_h, grid_w)} is not merge-aligned "
                f"to {merge_size}"
            )

        input_patches = grid_t * grid_h * grid_w
        merged_tokens = input_patches // (merge_size**2)

        if is_video:
            if len(timestamp_token_ids) != grid_t:
                raise ValueError(
                    "M3VL timestamp group count does not match temporal grid: "
                    f"{len(timestamp_token_ids)} != {grid_t}"
                )
            timestamp_tokens = sum(len(ids) for ids in timestamp_token_ids)
            output_tokens = merged_tokens + timestamp_tokens + 2 * grid_t
        else:
            output_tokens = merged_tokens + 2

        max_frames = self.visual.vision_config.vision_segment_max_frames
        if max_frames is None or max_frames <= 0:
            segment_frames = [grid_t]
        else:
            segment_frames = [
                min(max_frames, grid_t - start)
                for start in range(0, grid_t, max_frames)
            ]
        segment_lengths = [frames * grid_h * grid_w for frames in segment_frames]

        return MMWorkEstimate(
            input_patches=input_patches,
            output_tokens=output_tokens,
            estimated_workspace_bytes=(
                input_patches * self._workspace_bytes_per_patch()
            ),
            max_attention_segment=max(segment_lengths, default=0),
            attention_work=sum(length * length for length in segment_lengths),
        )

    def get_batch_work_budget(self, max_batch_media: int) -> Optional[MMWorkEstimate]:
        """Map the existing media cap to an M3VL-equivalent work budget."""
        # Serial mode passes sys.maxsize to preserve the historical unbounded
        # behavior. Cost admission is only useful for the bounded batch path.
        if max_batch_media >= 1 << 30:
            return None

        patch_size = int(self.mm_processor.patch_size)
        merge_size = int(self.merge_size)
        max_pixels = int(self.mm_processor.max_pixels)
        reference_patches = max(1, max_pixels // (patch_size**2))
        reference_output_tokens = reference_patches // (merge_size**2) + 2
        max_frames = self.visual.vision_config.vision_segment_max_frames
        max_segment_frames = max(1, int(max_frames or 1))

        reference = MMWorkEstimate(
            input_patches=reference_patches,
            output_tokens=reference_output_tokens,
            estimated_workspace_bytes=(
                reference_patches * self._workspace_bytes_per_patch()
            ),
            max_attention_segment=reference_patches * max_segment_frames,
            attention_work=reference_patches * reference_patches,
        )
        return reference.scaled(max_batch_media)

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "tokenizer": self.tokenizer,
            "image_token_index": self.image_token_index,
            "video_token_index": self.video_token_index,
            "video_fps": self.video_fps,
            "video_max_frames": self.video_max_frames,
            "video_min_frames": self.video_min_frames,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor=None,
        tokenizer=None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        **kwargs,
    ):
        assert (
            len(mm_inputs) == 1
        ), f"MiniMaxM3VL expects exactly one mm input per call, got {len(mm_inputs)}"
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type

        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            return MiniMaxM3VLImageEmbedding._preprocess_image(
                mm_input, vit_config, processor
            )
        elif mm_type == MMUrlType.VIDEO:
            return MiniMaxM3VLImageEmbedding._preprocess_video(
                mm_input, vit_config, processor, tokenizer, **kwargs
            )
        else:
            raise ValueError(f"unknown MMUrlType for MiniMaxM3VL: {mm_type}")

    @staticmethod
    def _preprocess_image(mm_input, vit_config, processor):
        """Download + decode only.  The resize/normalize/patch-fold transforms
        run later on GPU in ``embedding()`` — here we just decode to a raw uint8
        CHW tensor and precompute the (device-independent, cheap) target size
        from the per-request min/max pixels.

        Returns ``(raw_chw_uint8, (target_h, target_w), None)`` where the ``None``
        third element marks an image (no timestamps).
        """
        data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        try:
            image = Image.open(data).convert("RGB")
        except Exception:
            raise_mm(MMErr.IMG_OPEN)
        raw = torchvision.transforms.functional.pil_to_tensor(image)  # uint8 [C,H,W]

        _, height, width = raw.shape
        factor = processor.patch_size * processor.merge_size
        min_pixels = processor.min_pixels
        max_pixels = processor.max_pixels
        pre_cfg = mm_input.mm_preprocess_config
        if getattr(pre_cfg, "max_pixels", -1) > 0:
            max_pixels = int(pre_cfg.max_pixels)
        if getattr(pre_cfg, "min_pixels", -1) > 0:
            min_pixels = int(pre_cfg.min_pixels)
        max_long_side_pixel = getattr(pre_cfg, "max_long_side_pixel", -1)
        if max_long_side_pixel <= 0:
            max_long_side_pixel = getattr(processor, "max_long_side_pixel", None)

        target_h, target_w = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            min_image_dimension=vit_config.mm_image_min_dimension,
            max_image_aspect_ratio=vit_config.mm_image_max_aspect_ratio,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=MIN_SHORT_SIDE_PIXEL,
            max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
        )
        return raw, (target_h, target_w), None

    @staticmethod
    def _preprocess_video(mm_input, vit_config, processor, tokenizer, **kwargs):
        from decord import VideoReader, bridge

        bridge.set_bridge("torch")

        video_bytes = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_video_max_file_size_kb,
        )
        try:
            probe_vr = VideoReader(video_bytes, num_threads=0)
            total_frames = len(probe_vr)
            video_fps = float(probe_vr.get_avg_fps())
        except Exception:
            raise_mm(MMErr.VIDEO_INVALID)

        if total_frames <= 0 or video_fps <= 0 or not math.isfinite(video_fps):
            raise_mm(MMErr.VIDEO_INVALID)

        pre_cfg = mm_input.mm_preprocess_config
        target_fps = float(getattr(pre_cfg, "fps", 0))
        if not target_fps or target_fps <= 0:
            target_fps = float(kwargs.get("video_fps", 1.0))
        if (
            not math.isfinite(target_fps)
            or target_fps < MiniMaxM3VLImageEmbedding.MIN_VIDEO_FPS
            or target_fps > MiniMaxM3VLImageEmbedding.MAX_VIDEO_FPS
        ):
            raise_mm(
                MMErr.VIDEO_REQ.format(
                    f"fps must be in "
                    f"[{MiniMaxM3VLImageEmbedding.MIN_VIDEO_FPS}, "
                    f"{MiniMaxM3VLImageEmbedding.MAX_VIDEO_FPS}], got {target_fps}"
                )
            )
        max_frames = getattr(pre_cfg, "max_frames", 0)
        if not max_frames or max_frames <= 0:
            max_frames = kwargs.get("video_max_frames", 768)
        configured_max_frames = vit_config.mm_video_max_frames
        if configured_max_frames <= 0:
            configured_max_frames = VitConfig.DEFAULT_MM_VIDEO_MAX_FRAMES
        max_frames = min(int(max_frames), int(configured_max_frames))

        indices = compute_sampled_frame_indices(
            total_frames, video_fps, target_fps, int(max_frames)
        )
        if not indices:
            raise_mm(MMErr.VIDEO_REQ.format("no video frames can be sampled"))
        num_frames = len(indices)

        # Probe one sampled frame before batch decode so source-resolution frames
        # are never materialized for every sample at once.
        try:
            probe_frame = probe_vr[indices[0]]
            src_h, src_w = int(probe_frame.shape[0]), int(probe_frame.shape[1])
        except Exception:
            raise_mm(MMErr.VIDEO_INVALID)

        patch_size = processor.patch_size
        merge_size = kwargs.get("merge_size", 2)
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        factor = patch_size * merge_size

        max_long_side_pixel = getattr(pre_cfg, "max_long_side_pixel", -1)
        if max_long_side_pixel > 0:
            target_h, target_w = smart_resize(
                src_h,
                src_w,
                factor=factor,
                min_image_dimension=vit_config.mm_image_min_dimension,
                max_image_aspect_ratio=vit_config.mm_image_max_aspect_ratio,
                max_long_side_pixel=int(max_long_side_pixel),
                min_short_side_pixel=MIN_SHORT_SIDE_PIXEL,
                max_total_pixels=None,
            )
        else:
            # Preserve the released processor's legacy per-frame area behavior
            # when the new long-side control is not requested.
            frame_max_size = getattr(processor, "max_pixels", None)
            if frame_max_size and isinstance(frame_max_size, int):
                edge = int(math.sqrt(frame_max_size))
                target_w, target_h = get_hw_multiple_of((src_w, src_h), factor, edge)
            else:
                target_w, target_h = get_hw_multiple_of((src_w, src_h), factor, None)

        total_pixels = target_h * target_w * num_frames
        if total_pixels > VIDEO_MAX_TOTAL_PIXELS:
            raise_mm(
                MMErr.VIDEO_REQ.format(
                    f"video area {total_pixels} (width * height * frames) "
                    f"exceeds max_total_pixels {VIDEO_MAX_TOTAL_PIXELS} "
                    "after resizing"
                )
            )

        # Let Decord scale during decode so the sampled-frame tensor is bounded by
        # the model target size. Small sequential batches bound result-buffer
        # memory without forcing one seek/decode operation per frame.
        try:
            if (src_h, src_w) == (target_h, target_w):
                vr = probe_vr
            else:
                del probe_frame
                del probe_vr
                video_bytes.seek(0)
                vr = VideoReader(
                    video_bytes,
                    width=target_w,
                    height=target_h,
                    num_threads=0,
                )

            frames_tensor = torch.empty(
                (num_frames, target_h, target_w, 3), dtype=torch.uint8
            )
            for start in range(0, num_frames, _VIDEO_DECODE_CHUNK_FRAMES):
                end = min(start + _VIDEO_DECODE_CHUNK_FRAMES, num_frames)
                frame_chunk = vr.get_batch(indices[start:end])
                if tuple(frame_chunk.shape) != (end - start, target_h, target_w, 3):
                    raise ValueError(
                        f"unexpected decoded video chunk shape: {tuple(frame_chunk.shape)}"
                    )
                frames_tensor[start:end].copy_(frame_chunk)
            expected_shape = (num_frames, target_h, target_w, 3)
            if tuple(frames_tensor.shape) != expected_shape:
                raise ValueError(
                    f"unexpected decoded video shape: expected {expected_shape}, "
                    f"got {tuple(frames_tensor.shape)}"
                )
        except Exception:
            raise_mm(MMErr.VIDEO_INVALID)

        # grid_t after temporal padding (matches the fold done in _gpu_fold).
        pad_n = (
            temporal_patch_size - num_frames % temporal_patch_size
        ) % temporal_patch_size
        grid_t = (num_frames + pad_n) // temporal_patch_size

        # Compute per-temporal-group timestamp token IDs.
        timestamp_token_ids: List[List[int]] = []
        for gi in range(grid_t):
            raw_frame_idx = indices[min(gi * temporal_patch_size, len(indices) - 1)]
            ts = raw_frame_idx / video_fps
            ts_text = f"]<]{ts:.1f} seconds[>["
            ts_ids = tokenizer.encode(ts_text, add_special_tokens=False)
            timestamp_token_ids.append(ts_ids)

        return frames_tensor, (target_h, target_w), timestamp_token_ids

    def _lookup_word_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Index into the CPU word-embedding table and move result to GPU."""
        return self.word_embedding_weight[token_ids].to(
            device=self._device, dtype=self._data_type
        )

    def _assemble_image(self, vit_feats: torch.Tensor) -> torch.Tensor:
        """Prepend start-of-image and append end-of-image embeddings."""
        return torch.cat(
            [
                self._start_emb.unsqueeze(0),
                vit_feats,
                self._end_emb.unsqueeze(0),
            ],
            dim=0,
        )

    def _assemble_video(
        self,
        vit_feats: torch.Tensor,
        grid_thw: torch.Tensor,
        timestamp_token_ids: List[List[int]],
        ts_embs_cache: Optional[torch.Tensor] = None,
        ts_offset: int = 0,
    ) -> tuple:
        """Interleave timestamp / bracket / ViT embeddings per temporal group.

        Returns ``(assembled_tensor, new_ts_offset)`` so the caller can track
        position within a shared ``ts_embs_cache`` across items.
        """
        grid_t = grid_thw[0][0].item()
        grid_h = grid_thw[0][1].item()
        grid_w = grid_thw[0][2].item()
        merge_length = self.merge_size**2
        frame_seqlen = (grid_h * grid_w) // merge_length

        chunks: List[torch.Tensor] = []
        for gi in range(grid_t):
            # Timestamp word embeddings
            ts_ids = timestamp_token_ids[gi]
            n_ts = len(ts_ids)
            if ts_embs_cache is not None:
                ts_emb = ts_embs_cache[ts_offset : ts_offset + n_ts]
                ts_offset += n_ts
            else:
                ts_emb = self._lookup_word_embeddings(
                    torch.tensor(ts_ids, dtype=torch.long)
                )

            # ViT features for this temporal group
            start = gi * frame_seqlen
            group_feats = vit_feats[start : start + frame_seqlen]

            chunks.extend(
                [
                    ts_emb,
                    self._start_emb.unsqueeze(0),
                    group_feats,
                    self._end_emb.unsqueeze(0),
                ]
            )

        return torch.cat(chunks, dim=0), ts_offset

    def _gpu_fold(
        self,
        frames_nchw: torch.Tensor,
        target_hw,
        pixel_values_out: Optional[torch.Tensor] = None,
    ) -> tuple:
        """GPU resize + rescale + normalize + temporal-pad + patch-fold.

        Takes raw decoded frames ``[N, C, H, W]`` (N=1 for images) on any device
        and returns ``(pixel_values [total_patches, patch_dim], grid_thw [1,3])``.
        The fold math is identical to the previous CPU preprocess; only the device
        differs (torchvision bicubic runs on GPU when ``frames_nchw`` is on cuda).
        """
        device = self._device
        dtype = self._data_type
        p = self.mm_processor
        patch_size = p.patch_size
        merge_size = self.merge_size
        temporal_patch_size = self.temporal_patch_size
        target_h, target_w = target_hw

        # Transfer compact uint8 input first, then cast on GPU. Combining the
        # device and dtype conversion makes pageable H2D copies much slower.
        frames = frames_nchw.to(device=device).float()
        if tuple(frames.shape[-2:]) != (target_h, target_w):
            frames = torchvision.transforms.functional.resize(
                frames,
                [target_h, target_w],
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )

        video = frames.unsqueeze(0)  # (1, T, C, H, W)
        mean, std = self._ensure_preprocess_buffers_on_device()
        video.mul_(p.rescale_factor).sub_(mean).div_(std)

        T = video.shape[1]
        pad_n = (temporal_patch_size - T % temporal_patch_size) % temporal_patch_size
        if pad_n:
            padded = torch.empty(
                (video.shape[0], T + pad_n, *video.shape[2:]),
                device=video.device,
                dtype=video.dtype,
            )
            padded[:, :T].copy_(video)
            padded[:, T:].copy_(video[:, -1:].expand(-1, pad_n, -1, -1, -1))
            video = padded

        B, T_pad, channel, H, W = video.shape
        grid_t = T_pad // temporal_patch_size
        grid_h, grid_w = H // patch_size, W // patch_size

        patches = video.view(
            B,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values_shape = (
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        if pixel_values_out is None:
            pixel_values = torch.empty(
                pixel_values_shape,
                device=device,
                dtype=dtype,
            )
        else:
            if (
                pixel_values_out.shape != pixel_values_shape
                or pixel_values_out.device != device
                or pixel_values_out.dtype != dtype
                or not pixel_values_out.is_contiguous()
            ):
                raise ValueError(
                    "invalid M3VL packed pixel output: "
                    f"expected shape={pixel_values_shape} device={device} "
                    f"dtype={dtype}, got shape={tuple(pixel_values_out.shape)} "
                    f"device={pixel_values_out.device} dtype={pixel_values_out.dtype}"
                )
            pixel_values = pixel_values_out

        # Copy the strided folded view directly into the final BF16 layout. This
        # avoids a per-item FP32 flatten allocation followed by another cast and
        # lets batched_embedding provide slices of one packed destination.
        pixel_values.view(*patches.shape).copy_(patches)
        # Shape metadata stays on CPU. The vision tower creates cu_seqlens on
        # the target device once, avoiding a grid_thw D2H sync in every batch.
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
        return pixel_values, grid_thw

    @staticmethod
    def _raw_to_nchw(raw: torch.Tensor, is_video: bool) -> torch.Tensor:
        """Normalize a raw decoded input to ``[N, C, H, W]``.

        Image raw is ``[C, H, W]`` (from ``pil_to_tensor``); video raw is
        ``[N, H, W, C]`` (from decord ``get_batch``).
        """
        if is_video:
            return raw.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return raw.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    @torch.inference_mode()
    def embedding(self, data, mm_type=None, **kwargs):
        raw, target_hw, timestamp_token_ids = data
        device = self._device
        dtype = self._data_type

        self._ensure_bracket_embs_on_device()

        is_video = timestamp_token_ids is not None
        frames_nchw = self._raw_to_nchw(raw, is_video)
        pixel_values, grid_thw = self._gpu_fold(frames_nchw, target_hw)
        vit_feats = self._run_visual(pixel_values, grid_thw).to(dtype)

        if not is_video:
            result = self._assemble_image(vit_feats)
        else:
            result, _ = self._assemble_video(vit_feats, grid_thw, timestamp_token_ids)

        position_ids = torch.arange(result.shape[0], device=device, dtype=torch.int32)
        return result, position_ids

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        if len(data_list) <= 1:
            return super().batched_embedding(data_list, mm_types, **kwargs)

        device = self._device
        dtype = self._data_type
        self._ensure_bracket_embs_on_device()

        # --- 1. GPU transform (resize/normalize/fold) per item, then batch ViT ---
        all_thw: List[torch.Tensor] = []
        grids: List[torch.Tensor] = []
        split_sizes: List[int] = []

        # self.visual applies the spatial 2x2 merge (patch_merge_mlp), so its
        # output has pv.shape[0] // merge_size**2 rows per item. Split the batched
        # ViT output by the POST-merge count, not the pre-merge patch count.
        merge_length = self.merge_size**2
        patch_counts = [
            self.estimate_work(data, mm_type).input_patches
            for data, mm_type in zip(data_list, mm_types)
        ]
        patch_dim = (
            3
            * self.temporal_patch_size
            * self.mm_processor.patch_size
            * self.mm_processor.patch_size
        )
        batched_pv = torch.empty(
            (sum(patch_counts), patch_dim),
            device=device,
            dtype=dtype,
        )

        patch_offset = 0
        for (raw, target_hw, ts_info), patch_count in zip(data_list, patch_counts):
            frames_nchw = self._raw_to_nchw(raw, ts_info is not None)
            next_patch_offset = patch_offset + patch_count
            pv, thw = self._gpu_fold(
                frames_nchw,
                target_hw,
                pixel_values_out=batched_pv[patch_offset:next_patch_offset],
            )
            all_thw.append(thw)
            grids.append(thw)
            split_sizes.append(pv.shape[0] // merge_length)
            patch_offset = next_patch_offset

        batched_thw = torch.cat(all_thw, dim=0)
        all_vit_feats = self._run_visual(batched_pv, batched_thw).to(dtype)
        per_item_feats = torch.split(all_vit_feats, split_sizes, dim=0)

        # --- 2. Batch word-embedding lookup for all timestamps ---
        all_ts_ids: List[int] = []
        for _, _, ts_info in data_list:
            if ts_info is not None:
                for ids in ts_info:
                    all_ts_ids.extend(ids)

        ts_embs_all: Optional[torch.Tensor] = None
        if all_ts_ids:
            ts_tensor = torch.tensor(all_ts_ids, dtype=torch.long)
            ts_embs_all = self.word_embedding_weight[ts_tensor].to(
                device=device, dtype=dtype
            )

        # --- 3. Assemble per-item results ---
        results: List[tuple] = []
        ts_offset = 0

        for i, (_, _, ts_info) in enumerate(data_list):
            vit_feats = per_item_feats[i]

            if ts_info is None:
                emb = self._assemble_image(vit_feats)
            else:
                emb, ts_offset = self._assemble_video(
                    vit_feats,
                    grids[i],
                    ts_info,
                    ts_embs_cache=ts_embs_all,
                    ts_offset=ts_offset,
                )

            pos = torch.arange(emb.shape[0], device=device, dtype=torch.int32)
            results.append((emb, pos))

        return results


class MiniMaxM3VLVitWeight(BaseVitWeights):
    """
    Names every weight loaded from disk for the MiniMax-M3 VL vision stack.

    On disk (top-level of the safetensors index) the keys are bare:
        vision_tower.vision_model.*, multi_modal_projector.*, patch_merge_mlp.*
    so `_ckpt_prefix` is empty.

    Runtime names mirror the checkpoint except that each attention layer has a
    fused `qkv_proj`. MiniMaxM3VLDeployWeightInfo maps that live parameter to
    the three published q/k/v tensors and concatenates them while loading.
    """

    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part.visual."


class MiniMaxM3VLDeployWeightInfo(BaseMultiModalDeployWeightInfo):
    """Load published Q/K/V tensors into the fused runtime projection."""

    def get_weight_info(self):
        weights = []
        ckpt_prefix = self.vit_weights.ckpt_prefix
        for weight_name in self.vit_weights.weight_names:
            qkv_names = get_fused_qkv_checkpoint_names(weight_name)
            checkpoint_names = qkv_names or (weight_name,)
            weights.append(
                CustomAtomicWeight(
                    weight_name,
                    [
                        CkptWeightInfo(ckpt_prefix + name, identity)
                        for name in checkpoint_names
                    ],
                    concat_0 if qkv_names else identity,
                    split_func=sp_id,
                )
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class MiniMaxM3VLMixin(BaseMultiModalMixin):
    def _init_multimodal(self) -> None:
        self.mm_part = MiniMaxM3VLImageEmbedding(self.mm_related_params)

        # The live MiniMaxM3VLVisionTower hierarchy follows the checkpoint:
        #   visual.vision_tower.vision_model.*  /  visual.multi_modal_projector.*
        #   visual.patch_merge_mlp.*. The deploy weight info handles the fused
        # QKV exception.
        self.mm_related_params.vit_weights = MiniMaxM3VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls):
        return MiniMaxM3VLDeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        # Used only by eval_mm_model_size / eval_mm_model_param_count to
        # estimate device memory before the full mixin is instantiated.
        # Returning the vision tower (which contains all three trainable
        # sub-modules) is exactly the slice whose params we want to count.
        return MiniMaxM3VLImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["minimax_m3_vl"], MiniMaxM3VLMixin)
