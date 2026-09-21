"""V4.1 ViT/aligner adapter with row-major image spans and three delimiters."""

import json
from pathlib import Path

import torch
from torch import nn

from rtp_llm.models.multimodal.deepseek_v41_processor import (
    IMAGE,
    V41ImageInput,
    V41ImageProcessorConfig,
    V41PreparedInputs,
    image_token_types,
    num_image_tokens,
)
from rtp_llm.models.multimodal.deepseek_vision import Aligner, Attention, RMSNorm, ViT


class V41VisionAttention(Attention):
    def _apply_rotary(self, q, k, cos, sin):
        if q.is_cuda and not torch.is_grad_enabled():
            from rtp_llm.models_py.modules.dsv4._vision_rope_triton import (
                apply_vision_qk_rope,
            )

            result = apply_vision_qk_rope(q, k, cos, sin)
            if result is not None:
                return result
        return super()._apply_rotary(q, k, cos, sin)


class DeepSeekV41VisionEmbedding(nn.Module):
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
        vit_h, vit_w = image.n_vit_h, image.n_vit_w
        patches = image.patches.to(device=self._device, dtype=self._data_type)
        aligned = self.aligner(self.vision(patches, vit_h, vit_w), vit_h, vit_w)
        types = image.types.to(device=self._device)
        delimiters = torch.stack(
            (self.image_start, self.image_start, self.image_newline, self.image_end)
        )
        result = delimiters[types]
        result[types == IMAGE] = aligned
        return result.contiguous()

    @torch.inference_mode()
    def encode_prepared_images(self, images) -> list[torch.Tensor]:
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
        return [self.encode_image(image) for image in validated]

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
