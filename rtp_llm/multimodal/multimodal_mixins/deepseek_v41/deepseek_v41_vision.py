"""V4.1 ViT/aligner adapter with row-major image spans and three delimiters."""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from rtp_llm.metrics import GaugeMetrics
from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import (
    ResizeTables,
    prepare_resize_tables,
)
from rtp_llm.multimodal.mm_preprocess_result import (
    ImageSourceIdentity,
    MMPreprocessResult,
    is_single_image_input,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    _METRIC_TAGS,
    V41ImageProcessorConfig,
    _preprocess_rgb_on_device,
    _rgb_tensor,
    load_image_source,
    num_image_tokens,
    plan_image_grid,
    preprocess_image,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    Aligner,
    Attention,
    RMSNorm,
    ViT,
    _vision_cos_sin,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.gpu_input_transfer import (
    RGBBatchUploader,
    UploadedImage,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.vit_metrics import vit_preprocess_timer
from rtp_llm.utils.base_model_datatypes import MMUrlType


@dataclass(frozen=True)
class _DecodedImage:
    """CPU-only result of the generic media decoder inside the ViT backend."""

    rgb: torch.Tensor
    resize_tables: ResizeTables | None = None


@dataclass(frozen=True)
class _RuntimeImage:
    """Backend-only patches and grid consumed by V4.1 ViT/aligner."""

    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int


class V41VisionAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self._attention_backend = os.environ.get("V41_VISION_ATTN_BACKEND", "fa4")
        if self._attention_backend not in ("sdpa", "fa4"):
            raise ValueError("V41_VISION_ATTN_BACKEND must be sdpa or fa4")

    def _attention(self, q, k, v):
        if self._attention_backend == "fa4":
            from rtp_llm.models_py.modules.dsv41.vision_attention import (
                apply_vision_fa4,
            )

            result = apply_vision_fa4(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            if result is not None:
                return result.squeeze(0)
        return super()._attention(q, k, v)

    def _apply_packed_rotary(self, q, k, cos, sin):
        from .packed_rotary import apply_packed_vision_qk_rope

        return apply_packed_vision_qk_rope(q, k, cos, sin)

    def _apply_rotary(self, q, k, cos, sin):
        if q.is_cuda and not torch.is_grad_enabled():
            from rtp_llm.models_py.modules.dsv41._vision_rope_triton import (
                apply_vision_qk_rope,
            )

            result = apply_vision_qk_rope(q, k, cos, sin)
            if result is not None:
                return result
        return super()._apply_rotary(q, k, cos, sin)


class DeepSeekV41VisionEmbedding(nn.Module, MultiModalEmbeddingInterface):
    strict_work_budget = True
    supports_inspection_sources = True
    supports_remote_source_identity = True

    def __init__(
        self,
        config,
        *,
        device=None,
        max_batch_images=sys.maxsize,
        batch_max_bytes=0,
        graph_max_entries=4,
        graph_max_bytes=0,
        cache_reserved_bytes=2 * 1024**3,
    ):
        super().__init__()
        self.processor_config = V41ImageProcessorConfig.from_model_config(config)
        # CPU workers decode RGB; the GPU owner transforms and packs images.
        # Existing MMScheduler owns cross-request admission and cancellation.
        self._rgb_uploader = None
        if max_batch_images < 1:
            raise ValueError("V4.1 packed image budget must be positive")
        if (
            min(
                batch_max_bytes,
                graph_max_entries,
                graph_max_bytes,
                cache_reserved_bytes,
            )
            < 0
        ):
            raise ValueError("V4.1 resource budgets must be nonnegative")
        self._packed_max_images = max_batch_images
        self._batch_max_bytes = batch_max_bytes
        self._graph_max_bytes = graph_max_bytes
        self._cache_reserved_bytes = cache_reserved_bytes
        self._resource_budget = None
        self._resource_lock = threading.Lock()
        self._staging_max_bytes = 4 * 1024**3
        params = config.vision_parameters()
        # Conservative live intermediates for packed projection, QKV/rotary,
        # residuals and FFN, plus input patches. Backend scratch varies; this is
        # admission estimation, not a promise that CUDA allocations cannot fail.
        self._workspace_per_patch = 2 * (
            3 * params["vision_patch_size"] ** 2
            + 16 * params["vision_dim"]
            + 6 * params["vision_inter_dim"]
        )
        self._packed_stats_lock = threading.Lock()
        self._packed_batch_histogram = {}
        from .packed_graph import PackedVisionGraph

        self._packed_graph = PackedVisionGraph(
            self._packed_max_images,
            max_entries=graph_max_entries,
            max_bytes=graph_max_bytes,
        )
        self.vision = ViT(config.vision_parameters(), attention_cls=V41VisionAttention)
        self.aligner = Aligner(config.vision_parameters())
        hidden_size = config.text["hidden_size"]
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.to(device=device, dtype=torch.bfloat16)

    def _apply(self, fn, recurse=True):
        graph = getattr(self, "_packed_graph", None)
        if graph is not None:
            graph.clear()
        self._resource_budget = None
        self._rgb_uploader = None
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        # Weight reload requires the same quiescent model lifecycle as before.
        # No old-address graph or pending job may survive replacement weights.
        if self._packed_graph is not None:
            self._packed_graph.clear()
        self._resource_budget = None
        result = super().load_state_dict(*args, **kwargs)
        if self._device.type == "cuda":
            from rtp_llm.models_py.modules.dsv41.vision_preprocess_triton import (
                warmup_preprocess,
            )

            warmup_preprocess(
                self._device, patch_size=self.processor_config.vision_patch_size
            )
            self._warmup_packed_kernels()
        return result

    @torch.inference_mode()
    def _warmup_packed_kernels(self):
        """Warm native packed kernels before accepting requests."""
        from rtp_llm.models_py.modules.dsv41._vision_rope_triton import (
            apply_vision_qk_rope,
        )
        from rtp_llm.models_py.modules.dsv41.vision_residual_add import (
            apply_vision_residual_rmsnorm,
        )
        from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

        if (
            not self.vision.blocks
            or torch.cuda.get_device_capability(self._device)[0] != 10
        ):
            return
        block = self.vision.blocks[0]
        if (block.attn.n_heads, block.attn.head_dim) != (16, 64):
            return
        with cuda_graph_gate.operation(), torch.cuda.device(self._device):
            qkv = torch.zeros(8, 3072, device=self._device, dtype=torch.bfloat16)
            q, k, _ = [part.view(8, 16, 64) for part in qkv.chunk(3, -1)]
            cos = torch.ones(8, 1, 32, device=self._device)
            sin = torch.zeros_like(cos)
            apply_vision_qk_rope(q, k, cos, sin)
            x = torch.zeros(8, 1024, device=self._device, dtype=torch.bfloat16)
            apply_vision_residual_rmsnorm(x, x, block.norm1.weight, block.norm1.eps)
            if block.attn._attention_backend == "fa4":
                from rtp_llm.models_py.modules.dsv41.vision_attention import (
                    warmup_vision_fa4_varlen,
                )

                warmup_vision_fa4_varlen(self._device)
            torch.cuda.current_stream().synchronize()

    def _resolve_resource_budget(self):
        with self._resource_lock:
            if self._resource_budget is not None:
                return self._resource_budget
            if self._device.type == "cuda":
                from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

                with cuda_graph_gate.operation():
                    free, _ = torch.cuda.mem_get_info(self._device)
                self._staging_max_bytes = min(
                    4 * 1024**3, max(16 * 1024**2, free // 16)
                )
                # Resolve once AFTER weights are loaded. Reserve both current
                # and prefetched RGB, result cache, and 10% free-memory headroom.
                available = free - max(free // 10, 1024**3)
                available -= 2 * self._staging_max_bytes + self._cache_reserved_bytes
                if available <= 0:
                    raise RuntimeError(
                        "insufficient free memory for ViT resource budgets"
                    )
            else:
                # CPU/meta inspection does not initialize a CUDA context.
                available = 64 * 1024**3
            graph_bytes = self._graph_max_bytes or available // 2
            if self._packed_graph is None or not self._packed_graph.max_entries:
                graph_bytes = 0
            batch_bytes = self._batch_max_bytes or (available - graph_bytes)
            if batch_bytes <= 0 or batch_bytes + graph_bytes > available:
                raise ValueError("ViT batch and graph budgets exceed available memory")
            if self._packed_graph is not None:
                self._packed_graph.max_bytes = graph_bytes
            self._resource_budget = MMWorkEstimate(
                estimated_workspace_bytes=batch_bytes
            )
            logging.info(
                "V4.1 resources: image_cap=%d workspace_bytes=%d graph_bytes=%d "
                "graph_entries=%d staging_bytes_per_batch=%d",
                self._packed_max_images,
                batch_bytes,
                graph_bytes,
                self._packed_graph.max_entries if self._packed_graph else 0,
                self._staging_max_bytes,
            )
            return self._resource_budget

    def _graph_context(self):
        return tuple(
            getattr(block.attn, "_attention_backend", "sdpa")
            for block in self.vision.blocks
        ) + (self.training, "residual_rmsnorm_r4_bf16", "quack_concat")

    def _forward_packed(self, patches, grids, *, rotary=None, cu_seqlens=None):
        from .packed_vision import forward_packed

        return forward_packed(
            self.vision,
            self.aligner,
            patches,
            grids,
            rotary=rotary,
            cu_seqlens=cu_seqlens,
        )

    def _prepare_graph_forward(self, grids):
        # Own each table, including when >32 distinct geometries evict the
        # global rotary LRU. Replay must never reference an evicted allocation.
        rotary = [
            _vision_cos_sin(
                h, w, self.vision.rope_dim, self.vision.rope_theta, str(self._device)
            )
            for h, w in grids
        ]
        from .packed_vision import prepare_fa4_cu_seqlens

        cu_seqlens = None
        if (
            self.vision.blocks
            and self.vision.blocks[0].attn._attention_backend == "fa4"
        ):
            cu_seqlens = prepare_fa4_cu_seqlens(grids, self._device)
        forward = partial(self._forward_packed, rotary=rotary, cu_seqlens=cu_seqlens)
        # Conservative per-entry charge even for tables shared with other keys.
        forward.graph_owned_bytes = sum(
            tensor.numel() * tensor.element_size() for pair in rotary for tensor in pair
        )
        if cu_seqlens is not None:
            forward.graph_owned_bytes += cu_seqlens.numel() * cu_seqlens.element_size()
        return forward

    def run_embedding_idle_work(self):
        if self._packed_graph is None or self.training or self._device.type != "cuda":
            return False
        return self._packed_graph.build_one(
            self._forward_packed,
            context=self._graph_context(),
            prepare_forward=self._prepare_graph_forward,
        )

    @property
    def _device(self):
        return self.image_start.device

    @property
    def _data_type(self):
        return self.image_start.dtype

    @classmethod
    def from_model_weights(cls, config, global_weights):
        """Bind the framework-installed vision tensors without duplicate allocation."""
        with torch.device("meta"):
            model = cls(config, device="meta")
        model.requires_grad_(False)
        expected = model.state_dict()
        state = {}
        for name in expected:
            state[name] = global_weights["v41." + name]
            if (
                "norm" in name
                and name.endswith("weight")
                and state[name].dtype != torch.bfloat16
            ):
                raise ValueError(
                    "FlashInfer vision norm weights must be loaded as BF16"
                )
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    def load_checkpoint(self, checkpoint: str | Path) -> dict[str, int]:
        """Load real vision tensors into the selected backend's parameter dtype."""
        from safetensors import safe_open

        checkpoint = Path(checkpoint)
        with (checkpoint / "model.safetensors.index.json").open(
            encoding="utf-8"
        ) as reader:
            mapping = json.load(reader)["weight_map"]
        required = set(self.state_dict())
        state = {}
        for shard in sorted({mapping[name] for name in required}):
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as tensors:
                for name in sorted(required):
                    if mapping[name] == shard:
                        state[name] = tensors.get_tensor(name)
        self.load_state_dict(state, strict=True)
        norms = [
            module for module in self.vision.modules() if isinstance(module, RMSNorm)
        ]
        return {
            "weight_tensors": len(required),
            "fp32_norm_tensors": sum(m.weight.dtype == torch.float32 for m in norms),
            "bf16_norm_tensors": sum(m.weight.dtype == torch.bfloat16 for m in norms),
        }

    @torch.inference_mode()
    def encode_image(self, image: _RuntimeImage) -> torch.Tensor:
        vit_h, vit_w = image.n_vit_h, image.n_vit_w
        patches = image.patches.to(device=self._device, dtype=self._data_type)
        aligned = self.aligner(self.vision(patches, vit_h, vit_w), vit_h, vit_w)
        return self._assemble_image_span(aligned, image)

    def _assemble_image_span(self, aligned, image):
        ratio = self.processor_config.vision_downsample_ratio
        llm_h = (image.n_vit_h + ratio - 1) // ratio
        llm_w = (image.n_vit_w + ratio - 1) // ratio
        return self._assemble_canonical_span(aligned, llm_h, llm_w)

    def _assemble_canonical_span(self, aligned, llm_h, llm_w):
        hidden = self.image_start.numel()
        # Static slices avoid CUDA nonzero and its device-to-host synchronization.
        result = self.image_start.new_empty((num_image_tokens(llm_h, llm_w), hidden))
        result[0].copy_(self.image_start)
        result[-1].copy_(self.image_end)
        rows = result[1:-1].view(llm_h, llm_w + 1, hidden)
        rows[:, :llm_w].copy_(aligned.view(llm_h, llm_w, hidden))
        rows[:, llm_w].copy_(self.image_newline)
        return result

    def estimate_work(self, data, mm_type=None):
        data = self._image_group(data)
        grids = [self._vit_grid(image) for image in data]
        sizes = [height * width for height, width in grids]
        ratio = self.processor_config.vision_downsample_ratio
        output_tokens = sum(
            num_image_tokens((h + ratio - 1) // ratio, (w + ratio - 1) // ratio)
            for h, w in grids
        )
        rgb_bytes = sum(
            image.rgb.numel() * image.rgb.element_size()
            for image in data
            if isinstance(image, (_DecodedImage, UploadedImage))
        )
        return MMWorkEstimate(
            input_patches=sum(sizes),
            output_tokens=output_tokens,
            estimated_workspace_bytes=(
                sum(sizes) * self._workspace_per_patch
                + rgb_bytes
                + output_tokens * self.image_start.numel() * 2 * 4
            ),
            max_attention_segment=max(sizes, default=0),
            attention_work=sum(size * size for size in sizes),
        )

    def get_batch_work_budget(self, max_batch_media):
        return self._resolve_resource_budget()

    def packed_forward_stats(self):
        with self._packed_stats_lock:
            return {
                "enabled": True,
                "stats_enabled": True,
                "image_batch_histogram": dict(self._packed_batch_histogram),
            }

    def packed_graph_stats(self):
        if self._packed_graph is None:
            return {"enabled": False, "captures": 0, "replays": 0, "eager": 0}
        return {
            "enabled": self._packed_graph.max_entries > 0,
            **self._packed_graph.stats(),
        }

    def input_transfer_stats(self):
        if self._rgb_uploader is None:
            return {"batches": 0, "bytes": 0, "budget_fallbacks": 0}
        return self._rgb_uploader.stats()

    def prepare_embedding_batch(self, data_list, mm_types):
        if self._device.type != "cuda":
            return None
        if len(data_list) != len(mm_types) or any(
            t != MMUrlType.IMAGE for t in mm_types
        ):
            raise ValueError("V4.1 staging requires matching IMAGE inputs")
        groups = [self._image_group(data) for data in data_list]
        if not all(
            isinstance(image, _DecodedImage) for group in groups for image in group
        ):
            return None
        self._resolve_resource_budget()
        if self._rgb_uploader is None:
            self._rgb_uploader = RGBBatchUploader(self._device, self._staging_max_bytes)
        staged = self._rgb_uploader.upload(
            [[image.rgb for image in group] for group in groups],
            [isinstance(data, _DecodedImage) for data in data_list],
        )
        if staged is not None:
            for index, (original, uploaded) in enumerate(zip(groups, staged.data_list)):
                items = [
                    replace(dst, resize_tables=src.resize_tables)
                    for src, dst in zip(original, self._image_group(uploaded))
                ]
                staged.data_list[index] = (
                    items[0] if isinstance(uploaded, UploadedImage) else items
                )
        return staged

    @torch.inference_mode()
    def encode_images_packed(self, images) -> list[torch.Tensor]:
        """Encode independent images together, preserving image and span order.

        The caller owns stream dependencies for CUDA patches and completion.
        Each bounded chunk uses the production packed ViT/aligner path and owns
        independent assembled outputs for its callers.
        """
        from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

        images = list(images)
        if not images:
            return []
        budget = self._resolve_resource_budget()
        estimates = [self.estimate_work(image) for image in images]
        if any(not estimate.fits_within(budget) for estimate in estimates):
            raise ValueError("V4.1 image exceeds packed workspace budget")
        outputs = []
        offset = 0
        while offset < len(images):
            end, work = offset, MMWorkEstimate()
            while end < len(images) and end - offset < self._packed_max_images:
                candidate = work + estimates[end]
                if not candidate.fits_within(budget):
                    break
                work = candidate
                end += 1
            chunk = images[offset:end]
            grids = [(image.n_vit_h, image.n_vit_w) for image in chunk]

            def consume(aligned):
                if len(aligned) != len(chunk):
                    raise RuntimeError("Packed V4.1 aligner output count mismatch")
                return [
                    self._assemble_image_span(value, image)
                    for value, image in zip(aligned, chunk)
                ]

            with cuda_graph_gate.operation():
                patches = [
                    image.patches.to(device=self._device, dtype=self._data_type)
                    for image in chunk
                ]
                if self._packed_graph is None:
                    outputs.extend(consume(self._forward_packed(patches, grids)))
            if self._packed_graph is not None:
                outputs.extend(
                    self._packed_graph.run(
                        patches,
                        grids,
                        self._forward_packed,
                        consume,
                        context=self._graph_context(),
                        estimated_bytes=work.estimated_workspace_bytes,
                    )
                )
            with self._packed_stats_lock:
                count = len(chunk)
                self._packed_batch_histogram[count] = (
                    self._packed_batch_histogram.get(count, 0) + 1
                )
            offset = end
        return outputs

    @staticmethod
    def preprocess_input(
        mm_inputs,
        vit_config,
        *,
        processor_config,
        gpu_preprocess=True,
        capture_inspection_sources=False,
        deadline=None,
    ):
        """Decode one bounded work item in MMProcessEngine's CPU executor.

        GPU mode returns owned CPU RGB plus immutable CPU resize tables.
        The tables stay with the work item through asynchronous H2D staging;
        resize, pad, normalization and patch folding
        are submitted by the GPU-owning scheduler thread.
        """
        if not mm_inputs or any(
            media.mm_type != MMUrlType.IMAGE for media in mm_inputs
        ):
            raise ValueError("V4.1 preprocessing requires non-empty IMAGE inputs")
        # Legacy image hints are ignored; the model config owns preprocessing.
        results = []
        inspection_sources = []
        source_identity = None
        capture_identity = bool(
            (
                getattr(vit_config, "mm_remote_cache_enable", False)
                or os.environ.get("MM_EMB_BENCHMARK_TIMING") == "1"
            )
            and is_single_image_input(mm_inputs)
        )
        deadline = (
            min(deadline, time.monotonic() + 60)
            if deadline is not None
            else time.monotonic() + 60
        )
        for media in mm_inputs:
            with vit_preprocess_timer(
                GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, _METRIC_TAGS
            ):
                payload = load_image_source(
                    media.url,
                    vit_config.download_headers,
                    max_file_size_kb=vit_config.mm_image_max_file_size_kb,
                    deadline=deadline,
                )
            if capture_inspection_sources or capture_identity:
                # Hash the same owned buffer passed to Image.open below. Never
                # re-download a URL to derive identity or approve cached pixels.
                source = payload.getvalue()
                if capture_inspection_sources:
                    inspection_sources.append(source)
                if capture_identity:
                    source_identity = ImageSourceIdentity.from_bytes(source)
            try:
                with Image.open(payload) as image:
                    if gpu_preprocess:
                        rgb = _rgb_tensor(image)
                        _, _, target_h, target_w = plan_image_grid(
                            rgb.shape[1], rgb.shape[0], processor_config
                        )
                        tables = prepare_resize_tables(
                            tuple(rgb.shape[:2]),
                            (target_h, target_w),
                            patch_size=processor_config.vision_patch_size,
                        )
                        results.append(_DecodedImage(rgb, tables))
                    else:
                        patches, vit_h, vit_w, _, _ = preprocess_image(
                            image, processor_config
                        )
                        results.append(_RuntimeImage(patches, vit_h, vit_w))
            except OSError as error:
                from rtp_llm.multimodal.mm_error_messages import MMErr

                raise ValueError(MMErr.IMG_OPEN) from error
        # Preserve the original scalar contract; existing embedding interfaces
        # already accept ordered image groups for multi-media work items.
        data = results[0] if len(results) == 1 else results
        return (
            MMPreprocessResult(data, inspection_sources, source_identity)
            if capture_inspection_sources or capture_identity
            else data
        )

    def get_preprocess_params(self):
        # Picklable CPU-only values: do not pass self or CUDA state to workers.
        return {
            "processor_config": self.processor_config,
            "gpu_preprocess": self._device.type == "cuda",
        }

    def validate_inputs(self, mm_inputs):
        for media in mm_inputs:
            if media.mm_type != MMUrlType.IMAGE:
                raise ValueError("V4.1 supports IMAGE inputs only")

    @staticmethod
    def _image_group(data):
        if isinstance(data, (_DecodedImage, UploadedImage, _RuntimeImage)):
            return [data]
        return list(data)

    def _vit_grid(self, image):
        if not isinstance(image, (_DecodedImage, UploadedImage)):
            return image.n_vit_h, image.n_vit_w
        height, width = image.rgb.shape[:2]
        _, _, target_h, target_w = plan_image_grid(width, height, self.processor_config)
        patch = self.processor_config.vision_patch_size
        return target_h // patch, target_w // patch

    def _materialize_image(self, image):
        if not isinstance(image, (_DecodedImage, UploadedImage)):
            return image
        n_vit_h, n_vit_w = self._vit_grid(image)
        if not self.estimate_work(image).fits_within(self._resolve_resource_budget()):
            raise ValueError("V4.1 decoded image exceeds packed workspace budget")
        patch = self.processor_config.vision_patch_size
        with torch.profiler.record_function("v41_gpu_preprocess"):
            patches = _preprocess_rgb_on_device(
                image.rgb,
                (n_vit_h * patch, n_vit_w * patch),
                patch,
                self._device,
                **(
                    {"tables": image.resize_tables}
                    if image.resize_tables is not None
                    else {}
                ),
            )
        return _RuntimeImage(patches, n_vit_h, n_vit_w)

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        images = self._image_group(data)
        images = [self._materialize_image(image) for image in images]
        outputs = self.encode_images_packed(images)
        return (
            (
                outputs[0]
                if isinstance(data, (_DecodedImage, UploadedImage, _RuntimeImage))
                else outputs
            ),
            [],
            [],
        )

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        if len(data_list) != len(mm_types):
            raise ValueError("V4.1 batch data/type counts differ")
        if any(mm_type != MMUrlType.IMAGE for mm_type in mm_types):
            raise ValueError("V4.1 supports IMAGE inputs only")
        groups = [self._image_group(data) for data in data_list]
        groups = [
            [self._materialize_image(image) for image in group] for group in groups
        ]
        outputs = self.encode_images_packed(
            [image for group in groups for image in group]
        )
        if len(outputs) != sum(map(len, groups)):
            raise RuntimeError("Packed V4.1 embedding output count mismatch")
        results, offset = [], 0
        for data, group in zip(data_list, groups):
            end = offset + len(group)
            values = outputs[offset:end]
            results.append(
                (
                    (
                        values[0]
                        if isinstance(
                            data, (_DecodedImage, UploadedImage, _RuntimeImage)
                        )
                        else values
                    ),
                    [],
                    [],
                )
            )
            offset = end
        return results


__all__ = ["DeepSeekV41VisionEmbedding", "V41VisionAttention"]
