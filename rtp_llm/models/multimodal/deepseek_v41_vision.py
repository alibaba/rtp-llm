"""V4.1 ViT/aligner adapter with row-major image spans and three delimiters."""

import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn

from rtp_llm.models.multimodal.deepseek_v41_processor import (
    IMAGE,
    V41ImageInput,
    V41ImageProcessorConfig,
    V41PreparedInputs,
    image_token_types,
    load_image_bytes,
    preprocess_image,
)
from rtp_llm.models.multimodal.deepseek_vision import Aligner, RMSNorm, ViT


class DeepSeekV41VisionEmbedding(nn.Module):
    def __init__(self, config, *, device=None):
        super().__init__()
        self.processor_config = V41ImageProcessorConfig.from_model_config(config)
        self.vision = ViT(config.vision_parameters())
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

    def create_weight_info(self):
        from rtp_llm.models.multimodal.deepseek_vision_weight import (
            DeepSeekVisionWeights,
        )

        return DeepSeekVisionWeights(
            {
                "vision": self.vision,
                "aligner": self.aligner,
                "image_start": self.image_start,
                "image_newline": self.image_newline,
                "image_end": self.image_end,
            }
        )

    @classmethod
    def from_model_weights(cls, config, global_weights):
        """Bind the framework-installed vision tensors without duplicate allocation."""
        with torch.device("meta"):
            model = cls(config, device="meta")
        expected = model.state_dict()
        state = {}
        device = None
        for name, placeholder in expected.items():
            value = global_weights["v41." + name]
            if value.shape != placeholder.shape or value.dtype != placeholder.dtype:
                raise ValueError(
                    f"V4.1 installed vision tensor {name} has wrong shape/dtype"
                )
            if value.device.type == "meta" or (
                device is not None and value.device != device
            ):
                raise ValueError(
                    "V4.1 installed vision tensors must share a materialized device"
                )
            state[name], device = value, value.device
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
        missing = required - set(mapping)
        if missing:
            raise ValueError(f"missing V4.1 vision tensors: {sorted(missing)}")
        relevant = {
            name
            for name in mapping
            if name.startswith(("vision.", "aligner.", "image_"))
        }
        if relevant != required:
            raise ValueError(
                f"unexpected V4.1 vision tensors: {sorted(relevant - required)}"
            )
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
        if any(module.weight.dtype != torch.float32 for module in norms):
            raise ValueError("V4.1 vision RMSNorm parameters must remain FP32")
        return {"weight_tensors": len(required), "fp32_norm_tensors": len(norms)}

    @torch.inference_mode()
    def encode_image(self, image: V41ImageInput) -> torch.Tensor:
        if image.processor_identity != self.processor_config.identity:
            raise ValueError(
                "image span was prepared with a different processor config"
            )
        vit_h, vit_w = image.n_vit_h, image.n_vit_w
        if image.patches.shape != (
            vit_h * vit_w,
            3,
            self.processor_config.vision_patch_size,
            self.processor_config.vision_patch_size,
        ):
            raise ValueError("image patch shape does not match the recorded ViT grid")
        ratio = self.processor_config.vision_downsample_ratio
        llm_h, llm_w = (vit_h + ratio - 1) // ratio, (vit_w + ratio - 1) // ratio
        expected_types = image_token_types(llm_h, llm_w)
        if image.types.dtype != torch.int64 or not torch.equal(
            image.types.cpu(), expected_types
        ):
            raise ValueError(
                "image token types do not match the canonical row-major grid"
            )
        patches = image.patches.to(device=self._device, dtype=self._data_type)
        aligned = self.aligner(self.vision(patches, vit_h, vit_w), vit_h, vit_w)
        if aligned.shape != (llm_h * llm_w, self.image_start.numel()):
            raise ValueError("aligner rows do not match the image grid")
        types = image.types.to(device=self._device)
        delimiters = torch.stack(
            (self.image_start, self.image_start, self.image_newline, self.image_end)
        )
        result = delimiters[types]
        result[types == IMAGE] = aligned
        return result.contiguous()

    @torch.inference_mode()
    def image_embedding(self, images: Sequence[Image.Image]) -> list[torch.Tensor]:
        result = []
        for image in images:
            patches, vit_h, vit_w, llm_h, llm_w = preprocess_image(
                image, self.processor_config
            )
            prepared = V41ImageInput(
                0,
                patches,
                vit_h,
                vit_w,
                image_token_types(llm_h, llm_w),
                "",
                self.processor_config.identity,
            )
            result.append(self.encode_image(prepared))
        return result

    @torch.inference_mode()
    def encode_prepared_images(self, images) -> list[torch.Tensor]:
        result = []
        for record in images:
            image = V41ImageInput(
                record["start"],
                record["patches"],
                record["n_vit_h"],
                record["n_vit_w"],
                record["types"].to(dtype=torch.int64),
                record["content_sha256"],
                record["processor_identity"],
            )
            result.append(self.encode_image(image))
        return result

    @torch.inference_mode()
    def mm_process(self, image: Image.Image, **kwargs):
        return self.image_embedding([image])[0]

    @torch.inference_mode()
    def mm_embedding(self, url: str, mm_type, download_headers: str = "", **kwargs):
        from rtp_llm.utils.multimodal_util import MMUrlType, get_bytes_io_from_url

        if mm_type not in (MMUrlType.IMAGE, MMUrlType.DEFAULT):
            raise ValueError("V4.1 vision embedding accepts images only")
        data = load_image_bytes(
            {"url": url},
            url_loader=lambda value: get_bytes_io_from_url(
                value, download_headers
            ).getvalue(),
        )
        with Image.open(io.BytesIO(data)) as image:
            patches, vit_h, vit_w, llm_h, llm_w = preprocess_image(
                image, self.processor_config
            )
        prepared = V41ImageInput(
            0,
            patches,
            vit_h,
            vit_w,
            image_token_types(llm_h, llm_w),
            hashlib.sha256(data).hexdigest(),
            self.processor_config.identity,
        )
        return self.encode_image(prepared), None

    @torch.inference_mode()
    def inject_embeddings(
        self, embeddings: torch.Tensor, prepared: V41PreparedInputs
    ) -> torch.Tensor:
        """Inject complete image spans before expanding the HC residual streams."""
        if embeddings.ndim != 2 or embeddings.shape != (
            len(prepared.token_ids),
            self.image_start.numel(),
        ):
            raise ValueError(
                "V4.1 image injection requires unexpanded [tokens, hidden] embeddings"
            )
        previous_end = 0
        for image in prepared.images:
            end = image.start + image.length
            if image.start < previous_end or end > len(prepared.token_ids):
                raise ValueError("image spans overlap or exceed the canonical sequence")
            if any(
                token != self.processor_config.image_token_id
                for token in prepared.token_ids[image.start : end]
            ):
                raise ValueError("image spans must retain the canonical image token ID")
            if tuple(image.types.tolist()) != prepared.token_types[image.start : end]:
                raise ValueError("image token types differ from the prepared sequence")
            embeddings[image.start : end].copy_(self.encode_image(image).to(embeddings))
            previous_end = end
        return embeddings
