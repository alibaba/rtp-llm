"""DeepSeek-VL2 vision/projector modules with strict newloader loading."""

import contextlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


class VisionEncoderConfig:
    def __init__(
        self,
        model_name: str = "siglip_so400m_patch14_384",
        image_size: int = 384,
        patch_size: int = 14,
        width: int = 1152,
        layers: int = 27,
        heads: int = 16,
        mlp_ratio: float = 3.7362,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **_: Any,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("vision model_name must be a non-empty string")
        self.model_name = model_name
        self.image_size = _positive_int(image_size, "vision image_size")
        self.patch_size = _positive_int(patch_size, "vision patch_size")
        self.width = _positive_int(width, "vision width")
        self.layers = _positive_int(layers, "vision layers")
        self.heads = _positive_int(heads, "vision heads")
        self.mlp_ratio = _positive_float(mlp_ratio, "vision mlp_ratio")
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint


class MlpProjectorConfig:
    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        token_pooling: bool = False,
        **_: Any,
    ) -> None:
        if projector_type not in {
            "identity",
            "linear",
            "mlp_gelu",
            "downsample_mlp_gelu",
        }:
            raise ValueError(f"unsupported projector_type={projector_type!r}")
        self.projector_type = projector_type
        self.input_dim = _positive_int(input_dim, "projector input_dim")
        self.n_embed = _positive_int(n_embed, "projector n_embed")
        self.depth = _positive_int(depth, "projector depth")
        self.mlp_ratio = _positive_int(mlp_ratio, "projector mlp_ratio")
        self.downsample_ratio = _positive_int(
            downsample_ratio, "projector downsample_ratio"
        )
        if not isinstance(token_pooling, bool):
            raise TypeError("projector token_pooling must be a bool")
        self.token_pooling = token_pooling


class MlpProjector(nn.Module):
    def __init__(self, cfg: MlpProjectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.projector_type == "identity":
            modules: nn.Module = nn.Identity()
        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)
        elif cfg.projector_type == "mlp_gelu":
            items: list[nn.Module] = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, cfg.depth):
                items.extend((nn.GELU(), nn.Linear(cfg.n_embed, cfg.n_embed)))
            modules = nn.Sequential(*items)
        else:
            items = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * cfg.mlp_ratio,
                )
            ]
            for _ in range(1, cfg.depth - 1):
                items.extend(
                    (
                        nn.GELU(),
                        nn.Linear(
                            cfg.n_embed * cfg.mlp_ratio,
                            cfg.n_embed * cfg.mlp_ratio,
                        ),
                    )
                )
            items.extend(
                (
                    nn.GELU(),
                    nn.Linear(cfg.n_embed * cfg.mlp_ratio, cfg.n_embed),
                )
            )
            modules = nn.Sequential(*items)

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)
        self.layers = modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"projector input must have shape [batch, tokens, hidden], got {x.shape}"
            )
        if self.cfg.token_pooling:
            batch_size, token_count, channels = x.shape
            side = math.isqrt(token_count)
            if side * side != token_count:
                raise ValueError(
                    f"token_pooling requires a square token grid, got {token_count}"
                )
            x = x.view(batch_size, side, side, channels).permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            _, channels, h_patches, w_patches, _, _ = patches.size()
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)
            x = self.token_pooling_layer(patches)
        elif self.cfg.projector_type == "downsample_mlp_gelu":
            batch_size, token_count, input_dim = x.shape
            side = math.isqrt(token_count)
            if side * side != token_count:
                raise ValueError(
                    "downsample projector requires a square token grid, "
                    f"got {token_count}"
                )
            remainder = side % self.cfg.downsample_ratio
            pad = self.cfg.downsample_ratio - remainder if remainder else 0
            x = x.reshape(batch_size, side, side, input_dim)
            if pad:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)
            x = x.permute(0, 3, 1, 2)
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            ).permute(0, 2, 1)
        return self.layers(x)


class ImageTransform:
    def __init__(
        self,
        mean: Optional[tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ) -> None:
        if mean is None or std is None:
            raise ValueError("image normalization mean/std must not be None")
        transforms: list[Callable] = [T.ToTensor()]
        if normalize:
            transforms.append(T.Normalize(mean, std))
        self.transform = T.Compose(transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


def select_best_resolution(
    image_size: tuple[int, int],
    candidate_resolutions: Sequence[Sequence[int]],
) -> tuple[int, int]:
    if not candidate_resolutions:
        raise ValueError("candidate_resolutions must not be empty")
    original_width, original_height = image_size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"invalid image size {image_size}")

    best_fit: Optional[tuple[int, int]] = None
    max_effective_resolution = -1
    min_wasted_resolution = float("inf")
    for index, resolution in enumerate(candidate_resolutions):
        if (
            not isinstance(resolution, Sequence)
            or isinstance(resolution, (str, bytes))
            or len(resolution) != 2
        ):
            raise ValueError(
                f"candidate_resolutions[{index}] must contain width and height"
            )
        width = _positive_int(resolution[0], f"candidate_resolutions[{index}][0]")
        height = _positive_int(resolution[1], f"candidate_resolutions[{index}][1]")
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective = min(
            downscaled_width * downscaled_height,
            original_width * original_height,
        )
        wasted = width * height - effective
        if effective > max_effective_resolution or (
            effective == max_effective_resolution and wasted < min_wasted_resolution
        ):
            max_effective_resolution = effective
            min_wasted_resolution = wasted
            best_fit = (width, height)
    if best_fit is None:
        raise RuntimeError("failed to select an image resolution")
    return best_fit


class DeepSeekVLV2VisionModel(RtpModule):
    """Checkpoint-shaped vision, projector, and separator parameter tree."""

    _TIMM_MODEL_NAMES = {
        "siglip_so400m_patch14_384": "vit_so400m_patch14_siglip_384.webli",
    }

    def __init__(
        self,
        config: Mapping[str, Any],
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("DeepSeek-VL2 vision config must be a mapping")
        if (
            not isinstance(params_dtype, torch.dtype)
            or not params_dtype.is_floating_point
        ):
            raise TypeError("vision params_dtype must be a floating torch.dtype")

        raw_vision_config = config.get("vision_config", {})
        raw_projector_config = config.get("projector_config", {})
        if not isinstance(raw_vision_config, Mapping):
            raise TypeError("vision_config must be a mapping")
        if not isinstance(raw_projector_config, Mapping):
            raise TypeError("projector_config must be a mapping")
        self.vision_config = VisionEncoderConfig(**raw_vision_config)
        self.projector_config = MlpProjectorConfig(**raw_projector_config)

        if self.vision_config.model_name not in self._TIMM_MODEL_NAMES:
            raise ValueError(
                "DeepSeek-VL2 newloader supports only "
                "vision model_name='siglip_so400m_patch14_384'; "
                f"got {self.vision_config.model_name!r}"
            )
        if (
            self.vision_config.image_size != 384
            or self.vision_config.patch_size != 14
            or self.vision_config.width != 1152
            or self.vision_config.layers != 27
            or self.vision_config.heads != 16
            or not math.isclose(
                self.vision_config.mlp_ratio,
                3.7362,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "DeepSeek-VL2 SigLIP checkpoint requires image_size=384, "
                "patch_size=14, width=1152, layers=27, heads=16, and "
                "mlp_ratio=3.7362; got "
                f"image_size={self.vision_config.image_size}, "
                f"patch_size={self.vision_config.patch_size}, "
                f"width={self.vision_config.width}, "
                f"layers={self.vision_config.layers}, "
                f"heads={self.vision_config.heads}, and "
                f"mlp_ratio={self.vision_config.mlp_ratio}"
            )
        if self.vision_config.width != self.projector_config.input_dim:
            raise ValueError(
                f"vision width={self.vision_config.width} does not match "
                f"projector input_dim={self.projector_config.input_dim}"
            )
        tile_tag = config.get("tile_tag", "2D")
        if tile_tag != "2D":
            raise ValueError(
                "DeepSeek-VL2 newloader currently supports only tile_tag='2D'; "
                f"got {tile_tag!r}"
            )
        self.tile_tag = tile_tag
        global_view_pos = config.get("global_view_pos", "head")
        if global_view_pos not in {"head", "tail"}:
            raise ValueError(
                "global_view_pos must be either 'head' or 'tail', "
                f"got {global_view_pos!r}"
            )
        self.global_view_pos = global_view_pos

        with set_default_torch_dtype(params_dtype):
            self.vision = timm.create_model(
                self._TIMM_MODEL_NAMES[self.vision_config.model_name],
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )
            self.projector = MlpProjector(self.projector_config)
            embed_std = 1 / math.sqrt(self.projector_config.n_embed)
            self.image_newline = nn.Parameter(
                torch.randn(self.projector_config.n_embed) * embed_std
            )
            self.view_seperator = nn.Parameter(
                torch.randn(self.projector_config.n_embed) * embed_std
            )
        self.to(dtype=params_dtype)

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: name.startswith(("vision.", "projector.")) or name in {
            "image_newline",
            "view_seperator",
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=self.dtype)
        return self.projector(self.vision.forward_features(images))

    @property
    def device(self) -> torch.device:
        return next(self.vision.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vision.parameters()).dtype


@register_model("deepseek_vl2_vision")
class DeepSeekVLV2ForVisionEmbedding(DeepSeekVLV2VisionModel):
    def __init__(self, model_config: Mapping[str, Any], load_config: Any) -> None:
        super().__init__(model_config, load_config.compute_dtype)


def load_deepseek_vl2_vision(
    *,
    vision_config: Mapping[str, Any],
    model_path: str,
    compute_dtype: torch.dtype,
    device: str,
) -> DeepSeekVLV2VisionModel:
    model_config = dict(vision_config)
    model_config.update(
        {
            "model_type": "deepseek_vl2_vision",
            "model_path": model_path,
        }
    )
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
        return loader.load()


__all__ = [
    "DeepSeekVLV2ForVisionEmbedding",
    "DeepSeekVLV2VisionModel",
    "ImageTransform",
    "MlpProjector",
    "MlpProjectorConfig",
    "VisionEncoderConfig",
    "load_deepseek_vl2_vision",
    "select_best_resolution",
    "set_default_torch_dtype",
]
