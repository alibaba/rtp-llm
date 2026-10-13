"""V4.1 ViT/aligner adapter with row-major image spans and three delimiters."""

import json
import os
import threading
from pathlib import Path

import torch
from torch import nn

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    IMAGE,
    V41ImageInput,
    V41ImageProcessorConfig,
    V41PreparedInputs,
    image_token_types,
    num_image_tokens,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    Aligner,
    Attention,
    RMSNorm,
    ViT,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)


class V41VisionAttention(Attention):
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
    def __init__(self, config, *, device=None):
        super().__init__()
        self.processor_config = V41ImageProcessorConfig.from_model_config(config)
        self.vision = ViT(config.vision_parameters(), attention_cls=V41VisionAttention)
        self.aligner = Aligner(config.vision_parameters())
        hidden_size = config.text["hidden_size"]
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.to(device=device, dtype=torch.bfloat16)
        for module in self.vision.modules():
            if isinstance(module, RMSNorm):
                module.weight.data = module.weight.data.float()
        # Image embeddings depend only on the frozen weights plus the content
        # identity carried in every V41 request. Identical images (common in
        # replayed conversations and cache-key-aligned bursts) collapse to one
        # GPU forward instead of serially re-encoding under the v41 execution
        # lock; see vit_rpc_server._v41_execution.
        self._encode_cache = {}
        self._encode_cache_order = []
        self._encode_cache_lock = threading.Lock()
        self._encode_cache_capacity = max(
            0, int(os.environ.get("V41_IMAGE_ENCODE_CACHE_ITEMS", "16"))
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
        for name, placeholder in expected.items():
            state[name] = global_weights["v41." + name]
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    def load_checkpoint(self, checkpoint: str | Path) -> dict[str, int]:
        """Load only the real vision and delimiter tensors, retaining norm FP32."""
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
        return {"weight_tensors": len(required), "fp32_norm_tensors": len(norms)}

    @torch.inference_mode()
    def encode_image(self, image: V41ImageInput) -> torch.Tensor:
        cached = self._encode_image_cached(image)
        if cached is not None:
            return cached
        return self._encode_image_compute(image)

    def _encode_cache_key(self, image: V41ImageInput):
        # A v41 image embedding is a pure function of the frozen weights and
        # the image payload. content_sha256 identifies the exact patch bytes,
        # processor_identity the preprocessing contract, and the grid fixes
        # the token layout; no other request field reaches the forward pass.
        # An EMPTY content_sha256 is not a trusted identity: RPC/prepared-input
        # checks still admit it, and two different images with the same grid
        # and processor would alias to one cached embedding. Bypass the cache
        # entirely when the trusted content identity is absent.
        if not image.content_sha256:
            return None
        return (
            image.processor_identity,
            image.content_sha256,
            image.n_vit_h,
            image.n_vit_w,
        )

    def _encode_image_cached(self, image: V41ImageInput):
        if self._encode_cache_capacity <= 0:
            return None
        key = self._encode_cache_key(image)
        if key is None:
            return None
        with self._encode_cache_lock:
            if key in self._encode_cache:
                self._encode_cache_order.remove(key)
                self._encode_cache_order.append(key)
                return self._encode_cache[key]
        return None

    def _encode_image_store(self, key, result: torch.Tensor):
        if key is None or self._encode_cache_capacity <= 0:
            return
        with self._encode_cache_lock:
            if key in self._encode_cache:
                return
            self._encode_cache[key] = result
            self._encode_cache_order.append(key)
            while len(self._encode_cache_order) > self._encode_cache_capacity:
                evicted = self._encode_cache_order.pop(0)
                self._encode_cache.pop(evicted, None)

    @torch.inference_mode()
    def _encode_image_compute(self, image: V41ImageInput) -> torch.Tensor:
        vit_h, vit_w = image.n_vit_h, image.n_vit_w
        patches = image.patches.to(device=self._device, dtype=self._data_type)
        aligned = self.aligner(self.vision(patches, vit_h, vit_w), vit_h, vit_w)
        types = image.types.to(device=self._device)
        delimiters = torch.stack(
            (self.image_start, self.image_start, self.image_newline, self.image_end)
        )
        result = delimiters[types]
        result[types == IMAGE] = aligned
        result = result.contiguous()
        self._encode_image_store(self._encode_cache_key(image), result)
        return result

    @torch.inference_mode()
    def validate_prepared_images(self, images) -> list[V41ImageInput]:
        validated = []
        previous_end = 0
        processor = self.processor_config
        for record in images:
            height, width = record["n_vit_h"], record["n_vit_w"]
            if (
                type(height) is not int
                or type(width) is not int
                or min(height, width) <= 0
            ):
                raise ValueError("V4.1 image grid must contain positive integers")
            ratio = processor.vision_downsample_ratio
            llm_h, llm_w = (height + ratio - 1) // ratio, (width + ratio - 1) // ratio
            if num_image_tokens(llm_h, llm_w) > processor.vision_max_n_token:
                raise ValueError("V4.1 image exceeds the configured token limit")
            patches = record["patches"]
            patch_size = processor.vision_patch_size
            if (
                not isinstance(patches, torch.Tensor)
                or patches.dtype != torch.bfloat16
                or tuple(patches.shape) != (height * width, 3, patch_size, patch_size)
                or record["processor_identity"] != processor.identity
            ):
                raise ValueError("Invalid V4.1 prepared patches or processor identity")
            types = record["types"]
            expected_types = image_token_types(llm_h, llm_w)
            if (
                not isinstance(types, torch.Tensor)
                or types.device.type != "cpu"
                or types.dtype not in (torch.int32, torch.int64)
                or not torch.equal(types, expected_types)
            ):
                raise ValueError("Invalid V4.1 prepared image token types")
            start = record["start"]
            if type(start) is not int or start < previous_end:
                raise ValueError(
                    "V4.1 image spans must be nonnegative and non-overlapping"
                )
            previous_end = start + expected_types.numel()
            image = V41ImageInput(
                start,
                patches,
                height,
                width,
                types.to(dtype=torch.int64),
                record["content_sha256"],
                record["processor_identity"],
            )
            validated.append(image)
        # Validate the whole request before any image is copied to the GPU.
        return validated

    def encode_prepared_images(self, images) -> list[torch.Tensor]:
        return [
            self.encode_image(image) for image in self.validate_prepared_images(images)
        ]

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        raise ValueError("V4.1 images require typed prepared-image inputs")

    def embedding(self, data, **kwargs):
        return [self.encode_image(image) for image in data], [], []

    @torch.inference_mode()
    def inject_embeddings(
        self, embeddings: torch.Tensor, prepared: V41PreparedInputs
    ) -> torch.Tensor:
        """Inject complete image spans before expanding the HC residual streams."""
        for image in prepared.images:
            end = image.start + image.length
            embeddings[image.start : end].copy_(self.encode_image(image).to(embeddings))
        return embeddings


__all__ = ["DeepSeekV41VisionEmbedding", "V41VisionAttention"]
