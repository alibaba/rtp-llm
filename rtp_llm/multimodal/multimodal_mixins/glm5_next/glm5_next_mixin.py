import math
from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    get_bytes_io_from_url,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


def _ceil_to_factor(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _fit_aligned_size(
    frames: int,
    height: int,
    width: int,
    factor: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max_pixels < frames * factor * factor:
        raise ValueError("max_pixels is too small for one aligned vision patch")
    low, high = 1, height
    best = (factor, factor)
    while low <= high:
        content_height = (low + high) // 2
        content_width = max(1, math.floor(width * content_height / height))
        canvas = (
            _ceil_to_factor(content_height, factor),
            _ceil_to_factor(content_width, factor),
        )
        if frames * canvas[0] * canvas[1] <= max_pixels:
            best = canvas
            low = content_height + 1
        else:
            high = content_height - 1
    return best


def glm5_smart_resize(
    height: int,
    width: int,
    *,
    temporal_patch_size: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Return GLM's upward-aligned, aspect-preserving padded canvas."""
    frames = temporal_patch_size
    canvas = (
        _ceil_to_factor(height, factor),
        _ceil_to_factor(width, factor),
    )
    pixels = frames * canvas[0] * canvas[1]
    if pixels > max_pixels:
        return _fit_aligned_size(frames, height, width, factor, max_pixels)
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / (frames * height * width))
        canvas = (
            _ceil_to_factor(max(1, math.ceil(height * scale)), factor),
            _ceil_to_factor(max(1, math.ceil(width * scale)), factor),
        )
        if frames * canvas[0] * canvas[1] > max_pixels:
            return _fit_aligned_size(frames, height, width, factor, max_pixels)
    return canvas


def _resize_and_pad(image: Image.Image, canvas: tuple[int, int], upscale: bool):
    canvas_height, canvas_width = canvas
    scale = min(canvas_height / image.height, canvas_width / image.width)
    if not upscale:
        scale = min(scale, 1.0)
    content_size = (
        max(1, min(canvas_width, math.floor(image.width * scale))),
        max(1, min(canvas_height, math.floor(image.height * scale))),
    )
    if image.size != content_size:
        image = image.resize(content_size, Image.Resampling.BICUBIC)
    padded = Image.new("RGB", (canvas_width, canvas_height))
    padded.paste(image, (0, 0))
    return padded


class Glm5NextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(
            dtype
        ) * self.weight


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float):
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
        )
        self.q_norm = Glm5NextRMSNorm(self.head_dim, 1e-5)
        self.k_norm = Glm5NextRMSNorm(self.head_dim, 1e-5)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(q.dtype).unsqueeze(1)
        sin = emb.sin().to(q.dtype).unsqueeze(1)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat((q_rot * cos + self._rotate_half(q_rot) * sin, q_pass), -1)
        k = torch.cat((k_rot * cos + self._rotate_half(k_rot) * sin, k_pass), -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        sequence_lengths: List[int],
        rotary_freqs: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x).view(-1, 3, self.num_heads, self.head_dim).unbind(1)
        q, k = self._apply_rope(self.q_norm(q), self.k_norm(k), rotary_freqs)
        outputs = []
        offset = 0
        for length in sequence_lengths:
            next_offset = offset + length
            q_i, k_i, v_i = (
                value[offset:next_offset].transpose(0, 1).unsqueeze(0)
                for value in (q, k, v)
            )
            out = F.scaled_dot_product_attention(q_i, k_i, v_i)
            outputs.append(out.squeeze(0).transpose(0, 1))
            offset = next_offset
        return self.proj(torch.cat(outputs).reshape(x.shape))


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(self, x, sequence_lengths, rotary_freqs):
        x = x + self.attn(self.norm1(x), sequence_lengths, rotary_freqs)
        return x + self.mlp(self.norm2(x))


class Glm5NextPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.out_hidden_size
        intermediate_size = config.projection_intermediate_size
        self.limit = config.swiglu_limit
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_projection_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.post_projection_norm(self.proj(x)))
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


class Glm5NextVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            Glm5NextVisionBlock(config) for _ in range(config.depth)
        )
        self.post_layernorm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm5NextPatchMerger(config)
        head_dim = config.hidden_size // config.num_heads
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

    def _rotary_freqs(self, grid_thw: torch.Tensor) -> torch.Tensor:
        positions = []
        merge = self.spatial_merge_size
        for frames, height, width in grid_thw.tolist():
            h = torch.arange(height).view(height, 1).expand(-1, width)
            w = torch.arange(width).view(1, width).expand(height, -1)
            h = h.view(height // merge, merge, width // merge, merge)
            w = w.view(height // merge, merge, width // merge, merge)
            positions.append(
                torch.stack(
                    (h.permute(0, 2, 1, 3).flatten(), w.permute(0, 2, 1, 3).flatten()),
                    -1,
                ).repeat(frames, 1)
            )
        position_ids = torch.cat(positions).to(self.rotary_inv_freq.device)
        return (position_ids[..., None] * self.rotary_inv_freq).flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        x = self.patch_embed(pixel_values)
        rotary_freqs = self._rotary_freqs(grid_thw)
        sequence_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).tolist()
        for block in self.blocks:
            x = block(x, sequence_lengths, rotary_freqs)
        x = self.post_layernorm(x)
        merge = self.spatial_merge_size
        x = x.view(-1, merge, merge, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample(x).flatten(1)
        return self.merger(x)


class Glm5NextImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        config = dict(mm_related_params.config["vision_config"])
        config["swiglu_limit"] = mm_related_params.config["swiglu_limit"]
        self.visual = Glm5NextVisionModel(SimpleNamespace(**config))
        self.processor_config = mm_related_params.config["processor_config"]
        processor_config = self.processor_config["image_processor"]
        self.processor = Qwen2VLImageProcessor(
            do_resize=False,
            do_rescale=processor_config.get("do_rescale", True),
            image_mean=processor_config["image_mean"],
            image_std=processor_config["image_std"],
            patch_size=processor_config["patch_size"],
            temporal_patch_size=processor_config["temporal_patch_size"],
            merge_size=processor_config["merge_size"],
        )

    @property
    def _data_type(self):
        return self.visual.patch_embed.proj.weight.dtype

    @property
    def _device(self):
        return self.visual.patch_embed.proj.weight.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        processor_config,
    ):
        if len(mm_inputs) != 1:
            raise ValueError("GLM5-Next preprocessing expects one multimodal input")
        mm_input = mm_inputs[0]
        if mm_input.mm_type not in (MMUrlType.DEFAULT, MMUrlType.IMAGE):
            raise ValueError("GLM5-Next video preprocessing is not enabled yet")
        image = Image.open(
            get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        ).convert("RGB")
        patch_size = processor_config["patch_size"]
        temporal_patch_size = processor_config["temporal_patch_size"]
        merge_size = processor_config["merge_size"]
        factor = (
            patch_size * merge_size * processor_config.get("patch_expand_factor", 1)
        )
        token_pixels = temporal_patch_size * (patch_size * merge_size) ** 2
        min_pixels = processor_config["min_image_tokens"] * token_pixels
        max_pixels = processor_config["max_image_tokens"] * token_pixels
        canvas = glm5_smart_resize(
            image.height,
            image.width,
            temporal_patch_size=temporal_patch_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = _resize_and_pad(
            image,
            canvas,
            upscale=temporal_patch_size * image.height * image.width < min_pixels,
        )
        result = processor(images=image, return_tensors="pt", do_resize=False)
        return result["pixel_values"], result["image_grid_thw"]

    def get_preprocess_params(self):
        return {
            "processor": self.processor,
            "processor_config": self.processor_config["image_processor"],
        }

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device, dtype=self._data_type)
        grid_thw = data[1].to(self._device)
        return self.visual(pixel_values, grid_thw), None


class Glm5NextVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class Glm5NextMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Glm5NextImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Glm5NextVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Glm5NextImageEmbedding(mm_related_params).visual


register_multimodal_mixin("glm5_next", Glm5NextMixin)
