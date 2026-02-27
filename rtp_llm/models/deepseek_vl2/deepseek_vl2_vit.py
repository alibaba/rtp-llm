import contextlib
from typing import List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, repeat
from PIL import Image, ImageOps
from transformers.configuration_utils import PretrainedConfig

from rtp_llm.models.multimodal.multimodal_common import ImageEmbeddingInterface


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


class VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "siglip_large_patch16_384"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        **kwargs,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class ImageTransform(object):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [T.ToTensor()]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x


def select_best_resolution(image_size, candidate_resolutions):
    # used for cropping
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


class DeepSeekVLV2ImageEmbedding(ImageEmbeddingInterface):

    def __init__(
        self,
        mm_related_params: "VitParameters",
        model_config=None,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        ignore_id: int = -100,
    ):
        self.mm_related_params = mm_related_params
        self.config = model_config
        vision_config_dict = mm_related_params.config.get("vision_config", {})
        vision_config = VisionEncoderConfig(**vision_config_dict)
        self.vision_config = vision_config
        self.patch_size = vision_config.patch_size
        self.image_size = vision_config.image_size

        self.image_mean = image_mean
        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        with set_default_torch_dtype(torch.float16):
            self.vision = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )
            self.vision = self.vision.to(dtype=torch.get_default_dtype())

        projector_config_dict = mm_related_params.config.get("projector_config", {})
        projector_config = MlpProjectorConfig(**projector_config_dict)
        self.projector_config = projector_config
        self.projector = MlpProjector(projector_config)
        self.downsample_ratio = projector_config.downsample_ratio

        self.ignore_id = ignore_id

        self.tile_tag = mm_related_params.config.get("tile_tag", "2D")
        self.global_view_pos = mm_related_params.config.get("global_view_pos", "head")
        self.candidate_resolutions = mm_related_params.config.get(
            "candidate_resolutions", {}
        )

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed))
                * embed_std
            )
        else:
            raise ValueError(
                f"tile tag should be either 1D or 2D, but got {self.tile_tag}"
            )

    @property
    def _device(self):
        return next(self.vision.parameters()).device

    @torch.no_grad()
    def image_embedding(self, images: List[Image.Image]) -> torch.Tensor:
        assert len(images) == 1

        image = images[0]
        images_list = []
        best_width, best_height = select_best_resolution(
            image.size, self.candidate_resolutions
        )

        """process the global view"""
        global_view = ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            color=tuple(int(x * 255) for x in self.image_mean),
        )
        images_list.append(self.image_transform(global_view))
        """process the local views"""
        local_view = ImageOps.pad(
            image,
            (best_width, best_height),
            color=tuple(int(x * 255) for x in self.image_mean),
        )

        """record height / width crop num"""
        num_width_tiles, num_height_tiles = (
            best_width // self.image_size,
            best_height // self.image_size,
        )

        for i in range(0, best_height, self.image_size):
            for j in range(0, best_width, self.image_size):
                images_list.append(
                    self.image_transform(
                        local_view.crop(
                            (j, i, j + self.image_size, i + self.image_size)
                        )
                    )
                )

        tensor_images = (
            torch.stack(images_list, dim=0)
            .to(device=self._device)
            .to(dtype=self._data_type)
        )

        images_feature = self.vision.forward_features(tensor_images)

        images_embeds = self.projector(images_feature)

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        num_tiles_in_image = num_width_tiles * num_height_tiles
        tile_index = 0
        # [hw, D]
        global_features = images_embeds[tile_index]

        # [num_height_tiles * num_width_tiles, hw, D]
        local_features = images_embeds[
            tile_index + 1 : tile_index + 1 + num_tiles_in_image
        ]

        # ----------------- global view add newline -----------------
        # [hw, D] -> [h, w, D]
        global_features = global_features.view(h, w, n_dim)
        # [D]     -> [h, 1, D]
        new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
        # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
        global_features = torch.cat([global_features, new_lines_in_global], dim=1)
        # [h, w + 1, D] -> [h * (w + 1), D]
        global_features = global_features.view(-1, n_dim)

        # ----------------- local view add newline -----------------
        # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
        local_features = rearrange(
            local_features,
            "(th tw) (h w) d -> (th h) (tw w) d",
            th=num_height_tiles,
            tw=num_width_tiles,
            h=h,
            w=w,
        )

        # [D] -> [num_height_tiles * h, 1, D]
        new_lines_in_local = repeat(
            self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h
        )

        # [num_height_tiles * h, num_width_tiles * w + 1, D]
        local_features = torch.cat([local_features, new_lines_in_local], dim=1)

        # [num_height_tiles * h, num_width_tiles * w + 1, D]
        #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
        local_features = local_features.view(-1, n_dim)

        # ----------------- merge global and local tiles -----------------
        if self.global_view_pos == "head":
            global_local_features = torch.cat(
                [global_features, self.view_seperator[None, :], local_features], dim=0
            )
        else:
            global_local_features = torch.cat(
                [local_features, self.view_seperator[None, :], global_features], dim=0
            )

        # Return a list of tensors, one per image, to match the interface
        return [global_local_features]
