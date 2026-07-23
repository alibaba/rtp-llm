import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.weight_mapper import WeightsMapper

logger = logging.getLogger(__name__)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: tuple[int, ...]

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "Qwen3VLVisionConfig":
        if not isinstance(config, Mapping):
            raise TypeError("vision_config must be a mapping")
        hidden_act = config.get("hidden_act", "gelu_pytorch_tanh")
        if not isinstance(hidden_act, str):
            raise TypeError("vision hidden_act must be a string")
        raw_indexes = config.get("deepstack_visual_indexes", (5, 11, 17))
        if not isinstance(raw_indexes, Sequence) or isinstance(
            raw_indexes, (str, bytes)
        ):
            raise TypeError("deepstack_visual_indexes must be a sequence of integers")
        indexes = tuple(raw_indexes)
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indexes
        ):
            raise TypeError("deepstack_visual_indexes must contain only integers")

        result = cls(
            depth=_positive_int(config.get("depth", 24), "vision depth"),
            hidden_size=_positive_int(
                config.get("hidden_size", 1024), "vision hidden_size"
            ),
            hidden_act=hidden_act,
            intermediate_size=_positive_int(
                config.get("intermediate_size", 4096),
                "vision intermediate_size",
            ),
            num_heads=_positive_int(config.get("num_heads", 16), "vision num_heads"),
            in_channels=_positive_int(
                config.get("in_channels", 3), "vision in_channels"
            ),
            patch_size=_positive_int(config.get("patch_size", 16), "vision patch_size"),
            spatial_merge_size=_positive_int(
                config.get("spatial_merge_size", 2),
                "vision spatial_merge_size",
            ),
            temporal_patch_size=_positive_int(
                config.get("temporal_patch_size", 2),
                "vision temporal_patch_size",
            ),
            out_hidden_size=_positive_int(
                config.get("out_hidden_size", 2560),
                "vision out_hidden_size",
            ),
            num_position_embeddings=_positive_int(
                config.get("num_position_embeddings", 2304),
                "vision num_position_embeddings",
            ),
            deepstack_visual_indexes=indexes,
        )
        if result.hidden_size % result.num_heads:
            raise ValueError(
                f"vision hidden_size={result.hidden_size} must be divisible by "
                f"num_heads={result.num_heads}"
            )
        head_dim = result.hidden_size // result.num_heads
        if head_dim % 4:
            raise ValueError(
                f"vision head_dim={head_dim} must be divisible by 4 for 2D rotary "
                "embedding"
            )
        grid_side = math.isqrt(result.num_position_embeddings)
        if grid_side * grid_side != result.num_position_embeddings:
            raise ValueError("num_position_embeddings must form a square grid")
        if len(set(indexes)) != len(indexes):
            raise ValueError("deepstack_visual_indexes must not contain duplicates")
        if any(index < 0 or index >= result.depth for index in indexes):
            raise ValueError(
                "deepstack_visual_indexes must refer to existing vision blocks"
            )
        if result.hidden_act not in {"gelu", "gelu_pytorch_tanh"}:
            raise ValueError(
                f"Unsupported Qwen3-VL vision activation {result.hidden_act!r}"
            )
        return result


class Qwen3VisionPatchEmbed(RtpModule):
    def __init__(self, config: Qwen3VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        kernel_size = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
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
                "Qwen3-VL pixel values must have shape "
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
        return hidden_states.reshape(-1, self.hidden_size)


class Qwen3VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.float()
    key = key.float()
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query.to(query_dtype), key.to(key_dtype)


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
        logger.info("Qwen3-VL vision flash attention is unavailable: %s", exc)
    return _FLASH_ATTN_VARLEN


class Qwen3VisionAttention(RtpModule):
    def __init__(self, config: Qwen3VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=True,
            dtype=params_dtype,
        )
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
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
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)
        query, key = _apply_vision_rotary(query, key, rotary_cos, rotary_sin)

        flash_attn_varlen = (
            _resolve_flash_attn_varlen()
            if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)
            else None
        )
        if flash_attn_varlen is not None:
            output = flash_attn_varlen(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
            ).reshape(seq_length, -1)
        else:
            query_chunks = torch.split(query, segment_lengths, dim=0)
            key_chunks = torch.split(key, segment_lengths, dim=0)
            value_chunks = torch.split(value, segment_lengths, dim=0)
            outputs = []
            for query_chunk, key_chunk, value_chunk in zip(
                query_chunks, key_chunks, value_chunks
            ):
                output = F.scaled_dot_product_attention(
                    query_chunk.transpose(0, 1).unsqueeze(0),
                    key_chunk.transpose(0, 1).unsqueeze(0),
                    value_chunk.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                )
                outputs.append(output.squeeze(0).transpose(0, 1))
            output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(output)


class Qwen3VisionMLP(RtpModule):
    def __init__(self, config: Qwen3VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.linear_fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=params_dtype,
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = F.gelu(
            hidden_states,
            approximate="tanh" if self.hidden_act == "gelu_pytorch_tanh" else "none",
        )
        return self.linear_fc2(hidden_states)


class Qwen3VisionBlock(RtpModule):
    def __init__(self, config: Qwen3VLVisionConfig, params_dtype: torch.dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6, dtype=params_dtype)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6, dtype=params_dtype)
        self.attn = Qwen3VisionAttention(config, params_dtype)
        self.mlp = Qwen3VisionMLP(config, params_dtype)

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


class Qwen3VisionPatchMerger(RtpModule):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        params_dtype: torch.dtype,
        use_postshuffle_norm: bool,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6, dtype=params_dtype)
        self.linear_fc1 = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            dtype=params_dtype,
        )
        self.linear_fc2 = nn.Linear(
            self.hidden_size,
            config.out_hidden_size,
            bias=True,
            dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
        hidden_states = F.gelu(self.linear_fc1(hidden_states))
        return self.linear_fc2(hidden_states)


class Qwen3VLVisionTransformer(RtpModule):
    def __init__(self, vision_config: Mapping[str, Any], params_dtype: torch.dtype):
        super().__init__()
        config = Qwen3VLVisionConfig.from_mapping(vision_config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.num_grid_per_side = math.isqrt(config.num_position_embeddings)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self.patch_embed = Qwen3VisionPatchEmbed(config, params_dtype)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings,
            config.hidden_size,
            dtype=params_dtype,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen3VisionBlock(config, params_dtype) for _ in range(config.depth)]
        )
        self.merger = Qwen3VisionPatchMerger(
            config, params_dtype, use_postshuffle_norm=False
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VisionPatchMerger(config, params_dtype, use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )

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

    def _grid_values(self, grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
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
        if grid_thw.device != self.device:
            raise ValueError(
                f"grid_thw must be on the vision model device {self.device}, "
                f"got {grid_thw.device}"
            )

        values = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
        for value in values:
            t, h, w = value
            if t <= 0 or h <= 0 or w <= 0:
                raise ValueError(f"grid_thw values must be positive, got {value}")
            if h % self.spatial_merge_size or w % self.spatial_merge_size:
                raise ValueError(
                    f"grid {value} must align to spatial_merge_size="
                    f"{self.spatial_merge_size}"
                )
        return values

    def _rotary_positions(
        self, grid_values: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        position_ids = []
        merge_size = self.spatial_merge_size
        for t, h, w in grid_values:
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

    def _interpolated_position_embeddings(
        self, grid_values: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        index_lists: list[list[int]] = [[], [], [], []]
        weight_lists: list[list[float]] = [[], [], [], []]
        for _, h, w in grid_values:
            h_indexes = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_indexes = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_indexes.int()
            w_floor = w_indexes.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            delta_h = h_indexes - h_floor
            delta_w = w_indexes - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indexes = (
                (base_h[:, None] + w_floor[None]).flatten(),
                (base_h[:, None] + w_ceil[None]).flatten(),
                (base_h_ceil[:, None] + w_floor[None]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None]).flatten(),
            )
            weights = (
                ((1 - delta_h)[:, None] * (1 - delta_w)[None]).flatten(),
                ((1 - delta_h)[:, None] * delta_w[None]).flatten(),
                (delta_h[:, None] * (1 - delta_w)[None]).flatten(),
                (delta_h[:, None] * delta_w[None]).flatten(),
            )
            for index in range(4):
                index_lists[index].extend(indexes[index].tolist())
                weight_lists[index].extend(weights[index].tolist())

        index_tensor = torch.tensor(index_lists, dtype=torch.long, device=self.device)
        weight_tensor = torch.tensor(weight_lists, dtype=self.dtype, device=self.device)
        position_embeddings = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
        interpolated = (
            position_embeddings[0]
            + position_embeddings[1]
            + position_embeddings[2]
            + position_embeddings[3]
        )
        spatial_sizes = [h * w for _, h, w in grid_values]
        spatial_embeddings = interpolated.split(spatial_sizes)

        outputs = []
        merge_size = self.spatial_merge_size
        for position_embedding, (t, h, w) in zip(spatial_embeddings, grid_values):
            position_embedding = position_embedding.repeat(t, 1)
            position_embedding = (
                position_embedding.reshape(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            outputs.append(position_embedding)
        return torch.cat(outputs)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if hidden_states.device != self.device:
            raise ValueError(
                f"pixel_values must be on the vision model device {self.device}, "
                f"got {hidden_states.device}"
            )
        grid_values = self._grid_values(grid_thw)
        expected_patches = sum(t * h * w for t, h, w in grid_values)
        if hidden_states.shape[0] != expected_patches:
            raise ValueError(
                f"pixel_values contain {hidden_states.shape[0]} patches, but "
                f"grid_thw describes {expected_patches}"
            )

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self._interpolated_position_embeddings(
            grid_values
        )
        rotary_pos_emb = self._rotary_positions(grid_values)
        rotary_cos = rotary_pos_emb.cos().unsqueeze(1).repeat(1, 1, 2).float()
        rotary_sin = rotary_pos_emb.sin().unsqueeze(1).repeat(1, 1, 2).float()
        segment_lengths = tuple(h * w for t, h, w in grid_values for _ in range(t))
        max_seqlen = max(segment_lengths)
        cu_seqlens = torch.tensor(
            (0, *accumulate(segment_lengths)),
            dtype=torch.int32,
            device=self.device,
        )

        deepstack_features = []
        deepstack_index_map = {
            layer_index: merger_index
            for merger_index, layer_index in enumerate(self.deepstack_visual_indexes)
        }
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rotary_cos,
                rotary_sin,
                segment_lengths,
                max_seqlen,
            )
            merger_index = deepstack_index_map.get(layer_index)
            if merger_index is not None:
                deepstack_features.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )
        return self.merger(hidden_states), deepstack_features


def _vision_config_from_model(
    model_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(model_config, Mapping):
        raise TypeError("qwen3_vl_vision model_config must be a mapping")
    vision_config = model_config.get("vision_config")
    if not isinstance(vision_config, Mapping):
        raise TypeError("qwen3_vl_vision requires model_config.vision_config")
    return vision_config


class Qwen3VLForVisionEmbedding(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.visual.": "visual."},
    )

    def __init__(self, model_config: Mapping[str, Any], load_config: NewLoaderConfig):
        super().__init__()
        if load_config.tp_size != 1 or load_config.tp_rank != 0:
            raise ValueError("Qwen3-VL vision newloader is single-rank and replicated")
        self.visual = Qwen3VLVisionTransformer(
            _vision_config_from_model(model_config), load_config.compute_dtype
        )

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: name.startswith("model.visual.")

    def load_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]] | dict[str, torch.Tensor]
    ) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights

        def legacy_compatible_weights():
            target_dtype = self.visual.dtype
            for name, tensor in iterator:
                if tensor.is_floating_point():
                    # The established Qwen3-VL vision path stages every checkpoint
                    # tensor through FP16 before converting it to the runtime dtype.
                    # Preserve that numerical contract so newloader and legacy
                    # inference produce the same BF16/FP32 features and tokens.
                    tensor = tensor.to(torch.float16).to(target_dtype)
                yield name, tensor

        super().load_weights(self.WEIGHTS_MAPPER.apply(legacy_compatible_weights()))

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.visual(hidden_states, grid_thw)


def load_qwen3_vl_vision(
    vision_config: Mapping[str, Any],
    model_path: str,
    compute_dtype: torch.dtype,
    device: str,
) -> Qwen3VLVisionTransformer:
    model_config = {
        "model_type": "qwen3_vl_vision",
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


__all__ = [
    "Qwen3VLForVisionEmbedding",
    "Qwen3VLVisionTransformer",
    "load_qwen3_vl_vision",
]
