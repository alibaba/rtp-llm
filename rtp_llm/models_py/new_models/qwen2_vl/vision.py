import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
from numbers import Real
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule

logger = logging.getLogger(__name__)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Qwen2VLVisionConfig:
    depth: int
    embed_dim: int
    hidden_size: int
    hidden_act: str
    mlp_ratio: float
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "Qwen2VLVisionConfig":
        if not isinstance(config, Mapping):
            raise TypeError("vision_config must be a mapping")
        hidden_act = config.get("hidden_act", "quick_gelu")
        if not isinstance(hidden_act, str):
            raise TypeError("vision hidden_act must be a string")
        mlp_ratio = config.get("mlp_ratio", 4.0)
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, Real):
            raise TypeError("vision mlp_ratio must be a real number")
        result = cls(
            depth=_positive_int(config.get("depth", 32), "vision depth"),
            embed_dim=_positive_int(config.get("embed_dim", 1280), "vision embed_dim"),
            hidden_size=_positive_int(
                config.get("hidden_size", 3584), "vision hidden_size"
            ),
            hidden_act=hidden_act,
            mlp_ratio=float(mlp_ratio),
            num_heads=_positive_int(config.get("num_heads", 16), "vision num_heads"),
            in_channels=_positive_int(
                config.get("in_chans", config.get("in_channels", 3)),
                "vision in_channels",
            ),
            patch_size=_positive_int(config.get("patch_size", 14), "vision patch_size"),
            spatial_merge_size=_positive_int(
                config.get("spatial_merge_size", 2),
                "vision spatial_merge_size",
            ),
            temporal_patch_size=_positive_int(
                config.get("temporal_patch_size", 2),
                "vision temporal_patch_size",
            ),
        )
        if result.embed_dim % result.num_heads != 0:
            raise ValueError(
                f"vision embed_dim={result.embed_dim} must be divisible by "
                f"num_heads={result.num_heads}"
            )
        head_dim = result.embed_dim // result.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"vision head_dim={head_dim} must be divisible by 4 for 2D rotary "
                "embedding"
            )
        if not math.isfinite(result.mlp_ratio) or result.mlp_ratio <= 0:
            raise ValueError(
                f"vision mlp_ratio must be finite and positive, got {result.mlp_ratio}"
            )
        if result.hidden_act not in {"gelu", "gelu_new", "quick_gelu"}:
            raise ValueError(
                f"Unsupported Qwen2-VL vision activation {result.hidden_act!r}"
            )
        return result


class Qwen2VisionPatchEmbed(RtpModule):
    def __init__(self, config: Qwen2VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        kernel_size = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_width = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        if hidden_states.ndim != 2 or hidden_states.shape[1] != expected_width:
            raise ValueError(
                "Qwen2-VL pixel values must have shape "
                f"[num_patches, {expected_width}], got {tuple(hidden_states.shape)}"
            )
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=self.proj.weight.dtype))
        return hidden_states.reshape(-1, self.embed_dim)


class Qwen2VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_q_dtype = q.dtype
    original_k_dtype = k.dtype
    q_float = q.float()
    k_float = k.float()
    q = (q_float * cos) + (_rotate_half(q_float) * sin)
    k = (k_float * cos) + (_rotate_half(k_float) * sin)
    return q.to(original_q_dtype), k.to(original_k_dtype)


_FLASH_ATTN_VARLEN: Optional[Callable[..., torch.Tensor]] = None
_FLASH_ATTN_RESOLVED = False


def _resolve_flash_attn_varlen() -> Optional[Callable[..., torch.Tensor]]:
    global _FLASH_ATTN_RESOLVED, _FLASH_ATTN_VARLEN
    if _FLASH_ATTN_RESOLVED:
        return _FLASH_ATTN_VARLEN
    _FLASH_ATTN_RESOLVED = True
    try:
        from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

        if can_use_flash_attn():
            from flash_attn import flash_attn_varlen_func

            _FLASH_ATTN_VARLEN = flash_attn_varlen_func
    except (ImportError, OSError) as exc:
        logger.info("Qwen2-VL vision flash attention is unavailable: %s", exc)
    return _FLASH_ATTN_VARLEN


class Qwen2VisionAttention(RtpModule):
    def __init__(self, config: Qwen2VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv = nn.Linear(
            config.embed_dim,
            3 * config.embed_dim,
            bias=True,
            dtype=params_dtype,
        )
        self.proj = nn.Linear(
            config.embed_dim,
            config.embed_dim,
            bias=True,
            dtype=params_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        segment_lengths: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        q, k = _apply_vision_rotary(q, k, rotary_cos, rotary_sin)

        flash_attn_varlen = (
            _resolve_flash_attn_varlen()
            if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
            else None
        )
        if flash_attn_varlen is not None:
            output = flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
            ).reshape(seq_length, -1)
        else:
            q_chunks = torch.split(q, segment_lengths, dim=0)
            k_chunks = torch.split(k, segment_lengths, dim=0)
            v_chunks = torch.split(v, segment_lengths, dim=0)
            outputs = []
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                output = F.scaled_dot_product_attention(
                    q_chunk.transpose(0, 1).unsqueeze(0),
                    k_chunk.transpose(0, 1).unsqueeze(0),
                    v_chunk.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                )
                outputs.append(output.squeeze(0).transpose(0, 1))
            output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(output)


def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "quick_gelu":
        return x * torch.sigmoid(1.702 * x)
    if name == "gelu_new":
        return F.gelu(x, approximate="tanh")
    return F.gelu(x)


class Qwen2VisionMLP(RtpModule):
    def __init__(self, config: Qwen2VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        intermediate_size = int(config.embed_dim * config.mlp_ratio)
        if intermediate_size <= 0:
            raise ValueError("Qwen2-VL vision MLP has an invalid intermediate size")
        self.hidden_act = config.hidden_act
        self.fc1 = nn.Linear(
            config.embed_dim, intermediate_size, bias=True, dtype=params_dtype
        )
        self.fc2 = nn.Linear(
            intermediate_size, config.embed_dim, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_apply_activation(self.fc1(x), self.hidden_act))


class Qwen2VisionBlock(RtpModule):
    def __init__(self, config: Qwen2VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6, dtype=params_dtype)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6, dtype=params_dtype)
        self.attn = Qwen2VisionAttention(config, params_dtype)
        self.mlp = Qwen2VisionMLP(config, params_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        segment_lengths: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            rotary_cos,
            rotary_sin,
            segment_lengths,
            max_seqlen,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen2VisionPatchMerger(RtpModule):
    def __init__(self, config: Qwen2VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.hidden_size = config.embed_dim * (config.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(config.embed_dim, eps=1e-6, dtype=params_dtype)
        self.mlp = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                dtype=params_dtype,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_size,
                config.hidden_size,
                bias=True,
                dtype=params_dtype,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states).reshape(-1, self.hidden_size)
        return self.mlp(hidden_states)


class Qwen2VisionTransformer(RtpModule):
    def __init__(self, vision_config: Mapping[str, Any], params_dtype: torch.dtype):
        super().__init__()
        config = Qwen2VLVisionConfig.from_mapping(vision_config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen2VisionPatchEmbed(config, params_dtype)
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen2VisionBlock(config, params_dtype) for _ in range(config.depth)]
        )
        self.merger = Qwen2VisionPatchMerger(config, params_dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_device(self) -> torch.device:
        return self.device

    def _rotary_positions(
        self, grid_values: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        position_ids = []
        merge_size = self.spatial_merge_size
        for t, h, w in grid_values:
            if t <= 0 or h <= 0 or w <= 0:
                raise ValueError(f"grid_thw values must be positive, got {(t, h, w)}")
            if h % merge_size or w % merge_size:
                raise ValueError(
                    f"grid ({t}, {h}, {w}) must align to spatial_merge_size="
                    f"{merge_size}"
                )
            h_positions = torch.arange(h, device=self.device).unsqueeze(1).expand(-1, w)
            h_positions = (
                h_positions.reshape(
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            w_positions = torch.arange(w, device=self.device).unsqueeze(0).expand(h, -1)
            w_positions = (
                w_positions.reshape(
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            position_ids.append(
                torch.stack((h_positions, w_positions), dim=-1).repeat(t, 1)
            )
        indices = torch.cat(position_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_values)
        return self.rotary_pos_emb(max_grid_size)[indices].flatten(1)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"grid_thw must have shape [num_items, 3], got {tuple(grid_thw.shape)}"
            )
        if grid_thw.shape[0] == 0:
            raise ValueError("grid_thw must describe at least one image or video")
        if (
            grid_thw.dtype == torch.bool
            or grid_thw.is_floating_point()
            or grid_thw.is_complex()
        ):
            raise TypeError("grid_thw must use an integer dtype")
        if hidden_states.device != grid_thw.device:
            raise ValueError(
                "pixel_values and grid_thw must be on the same device, got "
                f"{hidden_states.device} and {grid_thw.device}"
            )
        if hidden_states.device != self.device:
            raise ValueError(
                f"pixel_values must be on the vision model device {self.device}, "
                f"got {hidden_states.device}"
            )
        grid_values = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
        rotary_pos_emb = self._rotary_positions(grid_values)
        rotary_cos = rotary_pos_emb.cos().unsqueeze(1).repeat(1, 1, 2).float()
        rotary_sin = rotary_pos_emb.sin().unsqueeze(1).repeat(1, 1, 2).float()
        expected_patches = sum(t * h * w for t, h, w in grid_values)
        if hidden_states.shape[0] != expected_patches:
            raise ValueError(
                f"pixel_values contain {hidden_states.shape[0]} patches, but "
                f"grid_thw describes {expected_patches}"
            )
        hidden_states = self.patch_embed(hidden_states)
        segment_lengths = tuple(h * w for t, h, w in grid_values for _ in range(t))
        max_seqlen = max(segment_lengths)
        cu_seqlens = torch.tensor(
            (0, *accumulate(segment_lengths)),
            device=hidden_states.device,
            dtype=torch.int32,
        )
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rotary_cos,
                rotary_sin,
                segment_lengths,
                max_seqlen,
            )
        return self.merger(hidden_states)


def _vision_config_from_model(
    model_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(model_config, Mapping):
        raise TypeError("qwen2_vl_vision model_config must be a mapping")
    vision_config = model_config.get("vision_config")
    if not isinstance(vision_config, Mapping):
        raise TypeError("qwen2_vl_vision requires model_config.vision_config")
    return vision_config


class Qwen2VLForVisionEmbedding(RtpModule):
    def __init__(self, model_config: Mapping[str, Any], load_config: NewLoaderConfig):
        super().__init__()
        if load_config.tp_size != 1 or load_config.tp_rank != 0:
            raise ValueError("Qwen2-VL vision newloader is single-rank and replicated")
        self.visual = Qwen2VisionTransformer(
            _vision_config_from_model(model_config), load_config.compute_dtype
        )

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: name.startswith("visual.")

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        return self.visual(hidden_states, grid_thw)


def load_qwen2_vl_vision(
    vision_config: Mapping[str, Any],
    model_path: str,
    compute_dtype: torch.dtype,
    device: str,
) -> Qwen2VisionTransformer:
    model_config = {
        "model_type": "qwen2_vl_vision",
        "model_path": model_path,
        "vision_config": dict(vision_config),
    }
    load_config = NewLoaderConfig(
        compute_dtype=compute_dtype,
        device=device,
        load_method=NewLoaderLoadMethod.SCRATCH,
    )
    loader = NewModelLoader(
        model_config=model_config,
        load_config=load_config,
        model_path=model_path,
    )
    with torch.device("cpu"):
        model = loader.load()
    return model.visual
