"""V4.1 ViT/aligner adapter with row-major image spans and three delimiters."""

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from rtp_llm.metrics import GaugeMetrics
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

    def __init__(self, config, *, device=None, max_batch_images=126):
        super().__init__()
        self.processor_config = V41ImageProcessorConfig.from_model_config(config)
        # CPU workers decode RGB; the GPU owner transforms and packs images.
        # Existing MMScheduler owns cross-request admission and cancellation.
        self._rgb_uploader = None
        # Retain the qualified 126-image/1,058,400-patch safety ceilings.
        # The common VitConfig may request a smaller media batch.
        if max_batch_images < 1:
            raise ValueError("V4.1 packed image budget must be positive")
        self._packed_max_images = min(max_batch_images, 126)
        self._packed_max_patches = 1058400
        self._packed_stats_lock = threading.Lock()
        self._packed_batch_histogram = {}
        from .packed_graph import PackedVisionGraph

        # The graph itself checks CUDA, full-batch shape and inference mode;
        # unsupported shapes and CPU execution retain the eager path.
        self._packed_graph = PackedVisionGraph(
            self._packed_max_images, self._packed_max_patches
        )
        self.vision = ViT(config.vision_parameters(), attention_cls=V41VisionAttention)
        self.aligner = Aligner(config.vision_parameters())
        hidden_size = config.text["hidden_size"]
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.to(device=device, dtype=torch.bfloat16)

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
        return MMWorkEstimate(
            input_patches=sum(sizes),
            output_tokens=sum(
                num_image_tokens(
                    (height + ratio - 1) // ratio,
                    (width + ratio - 1) // ratio,
                )
                for height, width in grids
            ),
            max_attention_segment=max(sizes, default=0),
            attention_work=sum(size * size for size in sizes),
        )

    def get_batch_work_budget(self, max_batch_media):
        return MMWorkEstimate(input_patches=self._packed_max_patches)

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
        return {"enabled": True, **self._packed_graph.stats()}

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
        if self._rgb_uploader is None:
            # Bounded pinned staging as in the qualified 126-image path.
            self._rgb_uploader = RGBBatchUploader(self._device, 4 * 1024**3)
        return self._rgb_uploader.upload(
            [[image.rgb for image in group] for group in groups],
            [isinstance(data, _DecodedImage) for data in data_list],
        )

    @torch.inference_mode()
    def encode_images_packed(self, images) -> list[torch.Tensor]:
        """Encode independent images together, preserving image and span order.

        The caller owns stream dependencies for CUDA patches and completion.
        Each bounded chunk uses the production packed ViT/aligner path and owns
        independent assembled outputs for its callers.
        """
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_vision import (
            forward_packed,
        )
        from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

        images = list(images)
        # Keep the capacity guard: an image that cannot fit must not leave
        # the chunking loop at the same offset forever.
        if any(
            image.n_vit_h * image.n_vit_w > self._packed_max_patches for image in images
        ):
            raise ValueError("V4.1 image exceeds packed patch budget")
        outputs = []
        offset = 0
        while offset < len(images):
            end, patch_count = offset, 0
            while end < len(images) and end - offset < self._packed_max_images:
                count = images[end].n_vit_h * images[end].n_vit_w
                if patch_count + count > self._packed_max_patches:
                    break
                patch_count += count
                end += 1
            chunk = images[offset:end]
            grids = [(image.n_vit_h, image.n_vit_w) for image in chunk]

            def forward(patches, grids):
                return forward_packed(self.vision, self.aligner, patches, grids)

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
                    outputs.extend(consume(forward(patches, grids)))
            if self._packed_graph is not None:
                outputs.extend(self._packed_graph.run(patches, grids, forward, consume))
            with self._packed_stats_lock:
                count = len(chunk)
                self._packed_batch_histogram[count] = (
                    self._packed_batch_histogram.get(count, 0) + 1
                )
            offset = end
        return outputs

    @staticmethod
    def preprocess_input(
        mm_inputs, vit_config, *, processor_config, gpu_preprocess=True
    ):
        """Decode one bounded work item in MMProcessEngine's CPU executor.

        GPU mode returns an owned CPU RGB tensor. Grid selection later uses
        CPU shape metadata; H2D, resize, pad, normalization and patch folding
        are submitted by the GPU-owning scheduler thread.
        """
        if not mm_inputs or any(
            media.mm_type != MMUrlType.IMAGE for media in mm_inputs
        ):
            raise ValueError("V4.1 preprocessing requires non-empty IMAGE inputs")
        # Legacy image hints are ignored; the model config owns preprocessing.
        results = []
        deadline = time.monotonic() + 60
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
            try:
                with Image.open(payload) as image:
                    if gpu_preprocess:
                        results.append(_DecodedImage(_rgb_tensor(image)))
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
        return results[0] if len(results) == 1 else results

    def prepare_inspection_inputs(
        self, mm_inputs, vit_config, *, deadline, cancellation_event
    ):
        from rtp_llm.ops import MultimodalInput

        self.validate_inputs(mm_inputs)
        sources = {}
        result = []
        for media in mm_inputs:
            if media.url not in sources:
                payload = load_image_source(
                    media.url,
                    vit_config.download_headers,
                    max_file_size_kb=vit_config.mm_image_max_file_size_kb,
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                )
                sources[media.url] = (
                    "data:application/octet-stream;base64,"
                    + base64.b64encode(payload.getvalue()).decode("ascii")
                )
            result.append(
                MultimodalInput(
                    sources[media.url],
                    media.mm_type,
                    torch.empty(0),
                    media.mm_preprocess_config,
                )
            )
        return result

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
        if n_vit_h * n_vit_w > self._packed_max_patches:
            raise ValueError("V4.1 decoded image exceeds packed patch budget")
        patch = self.processor_config.vision_patch_size
        with torch.profiler.record_function("v41_gpu_preprocess"):
            patches = _preprocess_rgb_on_device(
                image.rgb,
                (n_vit_h * patch, n_vit_w * patch),
                patch,
                self._device,
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
