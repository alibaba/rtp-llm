"""DeepSeek-V4 Flash Vision encoder and image-token layout."""

import math
from functools import lru_cache
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

from rtp_llm.utils.multimodal_util import get_bytes_io_from_url, vit_emb_cache_

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height,
    width,
    best_height,
    best_width,
    patch_size,
    downsample_ratio,
    max_n_token,
):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def preprocess_image(image: Image.Image, config):
    patch_size = config["vision_patch_size"]
    width, height = image.size
    max_wh_ratio = config.get("vision_max_wh_ratio")
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    min_pixels = config["vision_min_pixels"]
    if 0 < width * height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        width, height = int(width * ratio), int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        config["vision_downsample_ratio"],
        config["vision_max_n_token"],
    )
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = (
        torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        .permute(2, 0, 1)
        .div_(255)
    )
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def build_image_attention_spans(
    raw_spans, prefix_lengths, device, swa_window_size: int = 128
):
    """Convert generic feature spans to aligned DSV4 image-attention spans.

    A shallow prefix cut is valid because the raw SWA tail still contains the
    image start. The cache allocator caps deeper cuts to the block before the
    image; this check is the model-side contract guard for other reuse paths.
    """
    spans = raw_spans.to(device="cpu", dtype=torch.long).reshape(-1, 3).clone()
    spans[:, 1] += 3 - spans[:, 1] % COMPRESS_PAD_TO
    spans[:, 2] -= 1
    prefixes = prefix_lengths.reshape(-1).cpu().tolist()
    for request_idx, image_start, image_end in spans.tolist():
        prefix = prefixes[request_idx]
        if image_start < prefix - (swa_window_size - 1) and prefix <= image_end:
            raise RuntimeError(
                "DeepSeek-V4 prefix reuse ended too deep inside an image block: "
                f"request={request_idx} image=[{image_start}, {image_end}] "
                f"prefix={prefix}"
            )
    return spans.to(device=device)


@lru_cache(32)
def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: str):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = (
        torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    ).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(
            3 * config["vision_patch_size"] ** 2, config["vision_dim"]
        )

    def forward(self, x):
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["vision_dim"]
        self.n_heads = config["vision_n_heads"]
        self.head_dim = dim // self.n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        ).squeeze(0)
        return self.wo(out.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, inter = config["vision_dim"], config["vision_inter_dim"]
        self.w1 = nn.Linear(dim, 2 * inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["vision_dim"]
        self.norm1, self.attn = RMSNorm(dim), Attention(config)
        self.norm2, self.mlp = RMSNorm(dim), MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_dim = config["vision_dim"] // config["vision_n_heads"] // 2
        self.rope_theta = float(config.get("vision_rope_theta", 10000.0))
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config["vision_n_layers"])]
        )
        self.norm = RMSNorm(config["vision_dim"])

    def forward(self, patches, n_h, n_w):
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, str(x.device)
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config["vision_downsample_ratio"]
        in_dim = config["vision_dim"] * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config["hidden_size"])
        self.w2 = nn.Linear(config["hidden_size"], config["hidden_size"])

    def forward(self, x, n_h, n_w):
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        channels, height, width = x.shape
        x = (
            x.reshape(channels, height // ratio, ratio, width // ratio, ratio)
            .permute(1, 3, 0, 2, 4)
            .reshape(-1, channels * ratio * ratio)
        )
        return self.w2(F.gelu(self.w1(x)))


class DeepSeekV4VisionEmbedding(nn.Module):
    def __init__(self, mm_related_params, model_config):
        nn.Module.__init__(self)
        self.config = model_config
        self.mm_related_params = mm_related_params
        cfg = mm_related_params.config
        self.vision = ViT(cfg)
        self.aligner = Aligner(cfg)
        hidden_size = cfg["hidden_size"]
        self.image_start = nn.Parameter(torch.empty(hidden_size))
        self.image_end = nn.Parameter(torch.empty(hidden_size))
        self.image_newline = nn.Parameter(torch.empty(hidden_size))
        self.image_pad = nn.Parameter(torch.empty(hidden_size))

    @property
    def _device(self):
        return self.image_start.device

    @property
    def _data_type(self):
        return self.config.compute_dtype

    def _mm_preprocess(self, data, **kwargs):
        return Image.open(data).convert("RGB")

    @torch.inference_mode()
    def mm_embedding(
        self, url: str, mm_type, download_headers: str = "", **kwargs: Any
    ):
        configs = kwargs.get("configs")
        start_mod4 = int(getattr(configs, "image_block_start_mod4", -1))
        if not 0 <= start_mod4 < COMPRESS_PAD_TO:
            raise ValueError(
                "DeepSeek-V4 image input is missing image_block_start_mod4"
            )
        cache_key = ("deepseek_v4", url, start_mod4)
        cached = vit_emb_cache_.check_cache(cache_key)
        if cached is not None:
            return cached
        data = get_bytes_io_from_url(url, download_headers=download_headers)
        image = self._mm_preprocess(data, mm_type=mm_type, **kwargs)
        features = self.mm_process(image, mm_type=mm_type, **kwargs)
        result = features.to(self._data_type).contiguous(), None
        vit_emb_cache_.insert_cache(cache_key, result)
        return result

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        configs = kwargs.get("configs")
        start_mod4 = int(getattr(configs, "image_block_start_mod4", -1))
        if not 0 <= start_mod4 < COMPRESS_PAD_TO:
            raise ValueError(
                "DeepSeek-V4 image input is missing image_block_start_mod4"
            )
        return self.image_embedding([mm_input], start_pos=start_mod4)[0]

    @torch.inference_mode()
    def image_embedding(
        self, images: List[Image.Image], start_pos: int = 0
    ) -> List[torch.Tensor]:
        outputs = []
        cfg = self.mm_related_params.config
        for image in images:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(image, cfg)
            patches = patches.to(device=self._device, dtype=self._data_type)
            embeds = self.aligner(
                self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w
            )
            types, perm = build_image_block(n_llm_h, n_llm_w, start_pos)
            types, perm = types.to(self._device), perm.to(self._device)
            params = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            )
            block = params[types]
            block[types == IMAGE] = embeds[perm]
            outputs.append(block)
            start_pos += block.size(0)
        return outputs


class DeepSeekV4VisionWeights:
    ckpt_prefix = ""
    ft_prefix = "self.mm_part."

    def __init__(self, vision_parts):
        self.weight_names = []
        for name, part in vision_parts.items():
            if isinstance(part, nn.Module):
                self.weight_names.extend(
                    f"{name}.{weight_name}" for weight_name in part.state_dict()
                )
            elif isinstance(part, nn.Parameter):
                self.weight_names.append(name)
            else:
                raise TypeError(f"unsupported vision weight owner: {type(part)}")


__all__ = [
    "DeepSeekV4VisionEmbedding",
    "DeepSeekV4VisionWeights",
    "build_image_block",
    "build_image_attention_spans",
    "grid_tokens",
    "preprocess_image",
    "safe_resize",
]
