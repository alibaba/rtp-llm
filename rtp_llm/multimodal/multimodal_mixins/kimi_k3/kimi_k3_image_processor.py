"""Kimi-K3 image processor."""

import asyncio
import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig

# All renderers in a service process share its startup worker configuration.
_executor_lock = Lock()
_executor: ThreadPoolExecutor | None = None
_image_decode_lock = Lock()


def load_kimi_k3_media_config(checkpoint_path: str) -> Dict[str, Any]:
    path = Path(checkpoint_path) / "preprocessor_config.json"
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)["media_proc_cfg"]


def _get_kimi_k3_media_executor(max_workers: int) -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="kimi-k3-media"
            )
        return _executor


def shutdown_kimi_k3_media_executor() -> None:
    """Drain and reset the K3 media pool."""
    global _executor
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def _preflight_kimi_k3_image(
    url: str, config: VitConfig
) -> tuple[torch.Tensor, tuple[int, int]]:
    from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url

    try:
        data = get_bytes_io_from_url(
            url,
            config.download_headers,
            max_file_size_kb=config.mm_image_max_file_size_kb,
        )
    except FtRuntimeException as error:
        if error.exception_type == ExceptionType.MM_WRONG_FORMAT_ERROR:
            raise
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR, str(error)
        ) from error
    raw = data.getbuffer()
    try:
        # Serialize full-resolution decoding to bound temporary pixel storage.
        with _image_decode_lock, Image.open(BytesIO(raw)) as image:
            size = image.size
            image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR, "Image could not be decoded"
        ) from error
    return torch.frombuffer(raw, dtype=torch.uint8), size


def preflight_kimi_k3_images(
    urls: Sequence[str], vit_config: VitConfig
) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    """Load images once, enforcing the per-image byte limit before decoding."""
    max_workers = vit_config.mm_preprocess_max_workers
    if not urls:
        return [], []
    executor = _get_kimi_k3_media_executor(max_workers)
    results: list[tuple[torch.Tensor, tuple[int, int]]] = []
    futures = []
    try:
        for offset in range(0, len(urls), max_workers):
            futures = [
                executor.submit(_preflight_kimi_k3_image, url, vit_config)
                for url in urls[offset : offset + max_workers]
            ]
            results.extend(future.result() for future in futures)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return [tensor for tensor, _ in results], [size for _, size in results]


async def preflight_kimi_k3_images_async(
    urls: Sequence[str], vit_config: VitConfig
) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    """Load images off the event loop with the same limits as sync preflight."""
    max_workers = vit_config.mm_preprocess_max_workers
    if not urls:
        return [], []
    executor = _get_kimi_k3_media_executor(max_workers)
    loop = asyncio.get_running_loop()
    results: list[tuple[torch.Tensor, tuple[int, int]]] = []
    futures = []
    try:
        for offset in range(0, len(urls), max_workers):
            futures = [
                loop.run_in_executor(
                    executor, _preflight_kimi_k3_image, url, vit_config
                )
                for url in urls[offset : offset + max_workers]
            ]
            results.extend(await asyncio.gather(*futures))
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return [tensor for tensor, _ in results], [size for _, size in results]


def _navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: Optional[int],
) -> Dict[str, int]:
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = int(fixed_output_tokens)
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        assert token_height * merge_kernel_size <= patch_limit_on_one_side
        assert token_width * merge_kernel_size <= patch_limit_on_one_side
        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _normalize(x: np.ndarray, mean: np.ndarray, std_inv: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x -= mean
    x *= std_inv
    return x


def _navit_patchify(pixel_values: np.ndarray, patch_size: int) -> Dict[str, np.ndarray]:
    T, H, W, C = pixel_values.shape
    assert C == 3
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


def _chessboard(
    height: int,
    width: int,
    square_size: int,
    square_on_top_left: bool,
    white_value: int,
    gray_value: int,
) -> np.ndarray:
    """Create one background without retaining image-sized arrays globally."""
    y = np.arange(height)[:, None] // square_size
    x = np.arange(width)[None, :] // square_size
    gray_mask = (x + y) % 2 == (1 if square_on_top_left else 0)
    background = np.full((height, width, 3), white_value, dtype=np.uint8)
    background[gray_mask] = gray_value
    return background


def _fill_transparent_background(
    image: Image.Image, config: Optional[Dict[str, Any]]
) -> Image.Image:
    if config is None:
        return image.convert("RGB")
    if image.mode == "RGB":
        return image
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")

    rgba = np.asarray(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    if config["pattern"] == "chessboard":
        background = _chessboard(
            height,
            width,
            config["chessboard_square_size"],
            config["chessboard_square_on_top_left"],
            config["chessboard_white_value"],
            config["chessboard_gray_value"],
        )
    else:
        value = {"white": 255, "black": 0, "gray": 128}[config["pattern"]]
        background = np.full((height, width, 3), value, dtype=np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    result = alpha * rgba[:, :, :3] + (1.0 - alpha) * background
    return Image.fromarray(result.astype(np.uint8))


class KimiK3VisionProcessor(BaseImageProcessor):
    """Image-only NaViT processor for Kimi-K3."""

    model_type = "kimi_k3"
    model_input_names = ["pixel_values", "grid_thws"]

    def __init__(self, media_proc_cfg: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        cfg = copy.deepcopy(media_proc_cfg)
        cfg.setdefault("transparent_bg_fill_stage", "before_resize")
        if cfg["transparent_bg_fill_stage"] not in ("before_resize", "after_resize"):
            raise ValueError("unsupported transparent_bg_fill_stage")
        bg = cfg.get("transparent_bg_config")
        if bg is not None:
            # Optional defaults match the checkpoint's TransparentBgConfig dataclass.
            bg = (
                dict(
                    pattern="black",
                    chessboard_square_size=16,
                    chessboard_square_on_top_left=True,
                    chessboard_white_value=255,
                    chessboard_gray_value=200,
                )
                | bg
            )
        cfg["transparent_bg_config"] = bg
        self.media_proc_cfg = cfg

    @staticmethod
    def _coerce_image(media: Dict[str, Any]) -> Dict[str, Any]:
        media_type = media.get("type", "image")
        if media_type != "image":
            raise ValueError(
                f"KimiK3VisionProcessor is image-only; got media type "
                f"{media_type!r}. Video / audio inputs are not supported."
            )
        image = media.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"KimiK3VisionProcessor expects PIL.Image, got {type(image)}"
            )
        # Keep alpha/palette transparency until the resized image is composited
        # onto K3's configured background.
        return {"type": "image", "image": image}

    def _resize_config(self, image: Image.Image) -> Dict[str, int]:
        width, height = image.size
        return self.resize_config_for_size(width, height)

    def resize_config_for_size(self, width: int, height: int) -> Dict[str, int]:
        cfg = self.media_proc_cfg
        return _navit_resize_image(
            width,
            height,
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )

    def _image_to_np(self, image: Image.Image, resize_to: tuple) -> np.ndarray:
        bg_config = self.media_proc_cfg["transparent_bg_config"]
        # Only differs on semi-transparent pixels: compositing before resize
        # blends the chessboard into resampled edges, after keeps it crisp.
        if self.media_proc_cfg["transparent_bg_fill_stage"] == "before_resize":
            image = _fill_transparent_background(image, bg_config)
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
        else:
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
            image = _fill_transparent_background(image, bg_config)
        return np.asarray(image)

    def media_tokens_calculator(self, media: Dict[str, Any]) -> int:
        media = self._coerce_image(media)
        return self._resize_config(media["image"])["num_tokens"]

    def preprocess(
        self,
        medias: Union[Dict[str, Any], List[Dict[str, Any]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if not isinstance(medias, list):
            medias = [medias]
        if not medias:
            return BatchFeature(data={}, tensor_type=return_tensors)

        cfg = self.media_proc_cfg
        image_mean = np.array(cfg["image_mean"])
        image_std_inv = 1.0 / np.array(cfg["image_std"])

        per_image_tensors: List[Dict[str, np.ndarray]] = []
        for item in medias:
            item = self._coerce_image(item)
            resize_config = self._resize_config(item["image"])
            new_width = resize_config["new_width"]
            new_height = resize_config["new_height"]
            pad_width = resize_config["pad_width"]
            pad_height = resize_config["pad_height"]

            array = self._image_to_np(item["image"], (new_width, new_height))
            if pad_height or pad_width:
                array = np.pad(
                    array,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            array = np.expand_dims(array, axis=0)
            array = _normalize(array, image_mean, image_std_inv)
            per_image_tensors.append(_navit_patchify(array, cfg["patch_size"]))

        pixel_value_tensors = [
            torch.from_numpy(item["pixel_values"]) for item in per_image_tensors
        ]
        grid_thw_tensors = [
            torch.from_numpy(item["grid_thw"]).to(torch.int64).unsqueeze(0)
            for item in per_image_tensors
        ]
        if len(per_image_tensors) == 1:
            pixel_values = pixel_value_tensors[0]
            grid_thws = grid_thw_tensors[0]
        else:
            pixel_values = torch.cat(pixel_value_tensors)
            grid_thws = torch.cat(grid_thw_tensors)
        return BatchFeature(
            data={"pixel_values": pixel_values, "grid_thws": grid_thws},
            tensor_type=return_tensors,
        )

    @staticmethod
    def make_image_prompt(width: int, height: int) -> str:
        return (
            f"<|media_begin|>image {width}x{height}"
            f"<|media_content|><|media_pad|><|media_end|>"
        )
