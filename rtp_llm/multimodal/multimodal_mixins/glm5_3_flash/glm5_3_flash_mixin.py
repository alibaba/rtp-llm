import math
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

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

GLM53_MM_LAYOUT_MAGIC = -53530053
GLM53_DEFAULT_VIDEO_MAX_TOKENS = 30000
GLM53_DEFAULT_VIDEO_MAX_FRAMES = 2048


def glm5_sample_frame_indices(
    total_frames: int,
    fps: float,
    duration: float,
    *,
    target_fps: float = 2.0,
    max_frame_count: int = GLM53_DEFAULT_VIDEO_MAX_FRAMES,
    temporal_patch_size: int = 2,
) -> list[int]:
    """Sample GLM-5.3 video frames with the training-reference policy."""
    if total_frames <= 0:
        raise ValueError("GLM-5.3-Flash video must contain at least one frame")
    if fps <= 0:
        raise ValueError("GLM-5.3-Flash video source fps must be positive")
    if target_fps <= 0:
        raise ValueError("GLM-5.3-Flash requested video fps must be positive")
    if temporal_patch_size <= 0:
        raise ValueError("GLM-5.3-Flash temporal_patch_size must be positive")
    if max_frame_count < temporal_patch_size:
        raise ValueError(
            "GLM-5.3-Flash max_frames must be at least temporal_patch_size"
        )
    # The ViT consumes frames in complete temporal groups.  Round the budget
    # down so duplicating an odd tail can never exceed the caller's max_frames.
    max_frame_count = (
        int(max_frame_count) // temporal_patch_size * temporal_patch_size
    )

    max_frame_idx = total_frames - 1
    if duration <= 0:
        duration = round(max_frame_idx / fps) + 1
    extract_t = min(
        max(int(duration * target_fps), 1),
        max_frame_count,
    )

    duration_per_frame = 1 / fps
    max_second = int(duration)
    if total_frames < extract_t:
        frame_indices = [
            math.floor(i * total_frames / extract_t) for i in range(extract_t)
        ]
    else:
        frame_indices = []
        current_second = 0.0
        interval = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(total_frames):
            if frame_index * duration_per_frame >= current_second:
                current_second += interval
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break

    if len(frame_indices) < extract_t:
        start = frame_indices[0] if frame_indices else 0
        end = frame_indices[-1] if frame_indices else max_frame_idx
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(
            0, total_frames - 1, extract_t, dtype=int
        ).tolist()

    unique_indices = list(dict.fromkeys(int(index) for index in frame_indices))
    if len(unique_indices) % temporal_patch_size:
        unique_indices.extend(
            [unique_indices[-1]]
            * (temporal_patch_size - len(unique_indices) % temporal_patch_size)
        )
    return unique_indices


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
    frames: int | None = None,
) -> tuple[int, int]:
    """Return GLM's upward-aligned, aspect-preserving padded canvas."""
    frames = frames if frames is not None else temporal_patch_size
    frames = _ceil_to_factor(frames, temporal_patch_size)
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


class Glm53FlashRMSNorm(nn.Module):
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


class Glm53FlashVisionPatchEmbed(nn.Module):
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


class Glm53FlashVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
        )
        self.q_norm = Glm53FlashRMSNorm(self.head_dim, 1e-5)
        self.k_norm = Glm53FlashRMSNorm(self.head_dim, 1e-5)
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


class Glm53FlashVisionMLP(nn.Module):
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


class Glm53FlashVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Glm53FlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm53FlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Glm53FlashVisionAttention(config)
        self.mlp = Glm53FlashVisionMLP(config)

    def forward(self, x, sequence_lengths, rotary_freqs):
        x = x + self.attn(self.norm1(x), sequence_lengths, rotary_freqs)
        return x + self.mlp(self.norm2(x))


class Glm53FlashPatchMerger(nn.Module):
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


class Glm53FlashVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm53FlashVisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            Glm53FlashVisionBlock(config) for _ in range(config.depth)
        )
        self.post_layernorm = Glm53FlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm53FlashPatchMerger(config)
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


class Glm53FlashImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        config = dict(mm_related_params.config["vision_config"])
        config["swiglu_limit"] = mm_related_params.config["swiglu_limit"]
        self.visual = Glm53FlashVisionModel(SimpleNamespace(**config))
        self.processor_config = mm_related_params.config["processor_config"]
        self.ckpt_path = mm_related_params.config["ckpt_path"]
        self.special_token_ids = mm_related_params.config[
            "vision_special_token_ids"
        ]
        self._timestamp_tokenizer = None
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
        video_processor_config = self.processor_config["video_processor"]
        self.video_processor = Qwen2VLImageProcessor(
            do_resize=False,
            do_rescale=video_processor_config.get("do_rescale", True),
            image_mean=video_processor_config["image_mean"],
            image_std=video_processor_config["image_std"],
            patch_size=video_processor_config["patch_size"],
            temporal_patch_size=video_processor_config["temporal_patch_size"],
            merge_size=video_processor_config["merge_size"],
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
        video_processor,
        processor_config,
    ):
        if len(mm_inputs) != 1:
            raise ValueError("GLM-5.3-Flash preprocessing expects one multimodal input")
        mm_input = mm_inputs[0]
        if mm_input.mm_type not in (
            MMUrlType.DEFAULT,
            MMUrlType.IMAGE,
            MMUrlType.VIDEO,
        ):
            raise ValueError(f"unsupported GLM-5.3-Flash media type: {mm_input.mm_type}")
        media_config = (
            processor_config["video_processor"]
            if mm_input.mm_type == MMUrlType.VIDEO
            else processor_config["image_processor"]
        )
        patch_size = media_config["patch_size"]
        temporal_patch_size = media_config["temporal_patch_size"]
        merge_size = media_config["merge_size"]
        factor = (
            patch_size * merge_size * media_config.get("patch_expand_factor", 1)
        )
        token_pixels = temporal_patch_size * (patch_size * merge_size) ** 2
        min_pixels = media_config["min_image_tokens"] * token_pixels
        max_pixels = media_config["max_image_tokens"] * token_pixels
        request_min_pixels = int(mm_input.mm_preprocess_config.min_pixels)
        request_max_pixels = int(mm_input.mm_preprocess_config.max_pixels)
        if request_min_pixels > 0:
            min_pixels = request_min_pixels
        if request_max_pixels > 0:
            max_pixels = min(max_pixels, request_max_pixels)
        if min_pixels > max_pixels:
            raise ValueError(
                "GLM-5.3-Flash min_pixels must not exceed the effective max_pixels"
            )

        data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        if mm_input.mm_type == MMUrlType.VIDEO:
            if VideoReader is None:
                raise ImportError(
                    "decord is required for GLM-5.3-Flash video processing"
                )
            video_reader = VideoReader(data, ctx=cpu(0), num_threads=1)
            total_frames = len(video_reader)
            source_fps = float(video_reader.get_avg_fps())
            requested_fps = float(mm_input.mm_preprocess_config.fps)
            if requested_fps <= 0:
                requested_fps = float(
                    media_config.get(
                        "fps_interval", media_config.get("fps", 2.0)
                    )
                )
            request_max_frames = int(mm_input.mm_preprocess_config.max_frames)
            processor_max_frames = int(
                media_config.get(
                    "max_frame_count_dynamic", GLM53_DEFAULT_VIDEO_MAX_FRAMES
                )
            )
            server_max_frames = int(vit_config.mm_video_max_frames)
            max_frames = min(
                value
                for value in (
                    request_max_frames,
                    processor_max_frames,
                    server_max_frames,
                )
                if value > 0
            )
            indices = glm5_sample_frame_indices(
                total_frames,
                source_fps,
                total_frames / source_fps,
                target_fps=requested_fps,
                max_frame_count=max_frames,
                temporal_patch_size=temporal_patch_size,
            )
            raw_frames = video_reader.get_batch(indices).asnumpy()
            del video_reader

            video_max_tokens = min(
                int(media_config["max_image_tokens"]),
                GLM53_DEFAULT_VIDEO_MAX_TOKENS,
            )
            max_pixels = min(max_pixels, video_max_tokens * token_pixels)
            if min_pixels > max_pixels:
                raise ValueError(
                    "GLM-5.3-Flash video min_pixels exceeds the 30000-token cap"
                )
            frame_height, frame_width = raw_frames.shape[1:3]
            canvas = glm5_smart_resize(
                frame_height,
                frame_width,
                temporal_patch_size=temporal_patch_size,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                frames=len(indices),
            )
            upscale = len(indices) * frame_height * frame_width < min_pixels
            frames = [
                _resize_and_pad(Image.fromarray(frame).convert("RGB"), canvas, upscale)
                for frame in raw_frames
            ]
            result = video_processor.preprocess(
                videos=frames, return_tensors="pt", do_resize=False
            )
            timestamps = [
                int(indices[i] / source_fps)
                for i in range(0, len(indices), temporal_patch_size)
            ]
            return (
                result["pixel_values_videos"],
                result["video_grid_thw"],
                timestamps,
            )

        image = Image.open(data).convert("RGB")
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
            "video_processor": self.video_processor,
            "processor_config": self.processor_config,
        }

    def _encode_timestamp(self, seconds: int) -> list[int]:
        if self._timestamp_tokenizer is None:
            from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
                TokenizerFactory,
            )

            self._timestamp_tokenizer = TokenizerFactory.create(
                self.ckpt_path, self.ckpt_path, "glm5_3_flash"
            )
        return self._timestamp_tokenizer.encode(
            f"{float(seconds):.1f} seconds", add_special_tokens=False
        )

    @staticmethod
    def _layout_tensor(
        *, group_start: bool, prefix_ids: list[int], suffix_ids: list[int]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                GLM53_MM_LAYOUT_MAGIC,
                int(group_start),
                len(prefix_ids),
                len(suffix_ids),
                *prefix_ids,
                *suffix_ids,
            ],
            dtype=torch.int32,
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device, dtype=self._data_type)
        grid_thw = data[1].to(self._device)
        embeddings = self.visual(pixel_values, grid_thw)
        if len(data) == 2:
            layout = self._layout_tensor(
                group_start=True, prefix_ids=[], suffix_ids=[]
            )
            return [embeddings], None, [layout]

        timestamps = data[2]
        grid_t = int(grid_thw[0, 0].item())
        if grid_t <= 0 or embeddings.shape[0] % grid_t != 0:
            raise ValueError(
                "GLM-5.3-Flash video embedding count is not divisible by grid_t"
            )
        if len(timestamps) != grid_t:
            raise ValueError(
                "GLM-5.3-Flash timestamp count does not match temporal grid"
            )
        frame_embeddings = list(embeddings.chunk(grid_t, dim=0))
        layouts = []
        for index, timestamp in enumerate(timestamps):
            layouts.append(
                self._layout_tensor(
                    group_start=index == 0,
                    prefix_ids=[self.special_token_ids["image_start"]],
                    suffix_ids=[
                        self.special_token_ids["image_end"],
                        *self._encode_timestamp(timestamp),
                    ],
                )
            )
        return frame_embeddings, None, layouts


class Glm53FlashVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class Glm53FlashMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Glm53FlashImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Glm53FlashVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Glm53FlashImageEmbedding(mm_related_params).visual


register_multimodal_mixin("glm5_3_flash", Glm53FlashMixin)
