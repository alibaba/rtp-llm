"""DeepSeek-V4.1 image preprocessing and canonical image-span metadata.

The formulas and four token types follow HF revision
2bc89ac599031fa673cab993f1df02fc4a98c673, inference/image_processor.py.
"""

import base64
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
IMAGE_PLACEHOLDER = "<\uff5cdeepseek_image\uff5c>"


@dataclass(frozen=True)
class V41ImageProcessorConfig:
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936
    vision_max_wh_ratio: None = None
    image_token_id: int = 129264
    vocab_size: int = 129280
    max_seq_len: int = 1048576

    @classmethod
    def from_model_config(cls, config):
        vision = config.vision
        return cls(
            vision_patch_size=vision["patch_size"],
            vision_downsample_ratio=vision["downsample_ratio"],
            vision_max_n_token=vision["max_image_tokens"],
            vision_min_pixels=vision["min_pixels"],
            vision_max_wh_ratio=vision.get("max_wh_ratio"),
            image_token_id=config.image_token_id,
            vocab_size=config.text["vocab_size"],
            max_seq_len=config.text["max_position_embeddings"],
        )

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class V41ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor
    content_sha256: str
    processor_identity: str

    @property
    def length(self) -> int:
        return self.types.numel()


@dataclass(frozen=True)
class V41PreparedInputs:
    prompt: str
    token_ids: tuple[int, ...]
    token_types: tuple[int, ...]
    images: tuple[V41ImageInput, ...]

    def append_text(self, text, token_ids):
        token_ids = tuple(token_ids)
        return replace(
            self,
            prompt=self.prompt + text,
            token_ids=self.token_ids + tuple(token_ids),
            token_types=self.token_types + (TEXT,) * len(token_ids),
        )

    @property
    def image_mask(self) -> torch.Tensor:
        return torch.tensor(self.token_types, dtype=torch.int64, device="cpu") != TEXT

    @property
    def image_content_hashes(self) -> tuple[str, ...]:
        return tuple(image.content_sha256 for image in self.images)


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    ratio = height / width
    max_width = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_height = max_width * ratio
    cell = patch_size * downsample_ratio
    if max_width < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_height < 1.0:
        return cell, (max_n_token - 3) * cell
    scale = min(
        math.floor(max_width) * cell / width, math.floor(max_height) * cell / height
    )
    return (
        math.floor(height * scale / patch_size) * patch_size,
        math.floor(width * scale / patch_size) * patch_size,
    )


def plan_image_grid(width: int, height: int, config: V41ImageProcessorConfig):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image width and height must be positive integers")
    patch = config.vision_patch_size
    if width * height < config.vision_min_pixels:
        ratio = (config.vision_min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width = math.ceil(width / patch) * patch
    best_height = math.ceil(height / patch) * patch
    llm_h, llm_w = llm_grid(
        best_height, best_width, patch, config.vision_downsample_ratio
    )
    if num_image_tokens(llm_h, llm_w) > config.vision_max_n_token:
        best_height, best_width = solve_resize_ratio(
            height,
            width,
            patch,
            config.vision_downsample_ratio,
            config.vision_max_n_token,
        )
        llm_h, llm_w = llm_grid(
            best_height, best_width, patch, config.vision_downsample_ratio
        )
    return llm_h, llm_w, best_height, best_width


def preprocess_image(image: Image.Image, config: V41ImageProcessorConfig):
    image = image.convert("RGB")
    llm_h, llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, config
    )
    patch = config.vision_patch_size
    vit_h, vit_w = best_height // patch, best_width // patch
    image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    values = (
        torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    )
    values = ((values - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        values.reshape(3, vit_h, patch, vit_w, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(vit_h * vit_w, 3, patch, patch)
    )
    return patches, vit_h, vit_w, llm_h, llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = (
        [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
    )
    return torch.tensor(types, dtype=torch.int64, device="cpu")


def load_image_bytes(
    record: Mapping[str, Any], *, url_loader: Callable[[str], bytes] | None = None
) -> bytes:
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data, validate=True)
    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"], validate=True)
        if source.get("url"):
            return load_image_bytes({"url": source["url"]}, url_loader=url_loader)
    url = record.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("image record has no supported data or URL source")
    if url.startswith("data:"):
        header, separator, payload = url.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("image data URLs require base64 encoding")
        return base64.b64decode(payload, validate=True)
    if url_loader is not None:
        return url_loader(url)
    from rtp_llm.utils.multimodal_util import get_bytes_io_from_url

    return get_bytes_io_from_url(url).getvalue()


def prepare_vl_inputs(
    prompt: str,
    images: Sequence[Mapping[str, Any]],
    tokenizer,
    config: V41ImageProcessorConfig,
    *,
    url_loader=None,
    output_budget: int = 0,
) -> V41PreparedInputs:
    # output_budget is retained for callers; the generation layer applies limits.
    return prepare_vl_inputs_from_token_ids(
        tokenizer.encode(prompt), images, config, prompt=prompt, url_loader=url_loader
    )


def prepare_vl_inputs_from_token_ids(
    prompt_tokens: Sequence[int],
    images: Sequence[Mapping[str, Any]],
    config: V41ImageProcessorConfig,
    *,
    prompt: str = "",
    url_loader=None,
    max_image_bytes: int | None = None,
) -> V41PreparedInputs:
    """Expand upstream image markers without re-tokenizing any text or special IDs."""
    if sum(token == config.image_token_id for token in prompt_tokens) != len(images):
        raise ValueError(
            "image placeholder count does not match the supplied image records"
        )
    tokens, token_types, image_inputs = [], [], []
    records = iter(images)
    for token in prompt_tokens:
        if token != config.image_token_id:
            tokens.append(token)
            token_types.append(TEXT)
            continue
        data = load_image_bytes(next(records), url_loader=url_loader)
        if max_image_bytes is not None and len(data) > max_image_bytes:
            raise ValueError("Multimodal file size is too large")
        try:
            with Image.open(io.BytesIO(data)) as image:
                patches, vit_h, vit_w, llm_h, llm_w = preprocess_image(image, config)
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("invalid or damaged V4.1 image payload") from error
        types = image_token_types(llm_h, llm_w)
        image_inputs.append(
            V41ImageInput(
                len(tokens),
                patches,
                vit_h,
                vit_w,
                types,
                hashlib.sha256(data).hexdigest(),
                config.identity,
            )
        )
        tokens.extend([config.image_token_id] * types.numel())
        token_types.extend(types.tolist())
    return V41PreparedInputs(
        prompt, tuple(tokens), tuple(token_types), tuple(image_inputs)
    )
