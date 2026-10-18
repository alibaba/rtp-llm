"""DeepSeek-V4.1 image preprocessing owned by the ViT backend.

The resize and grid formulas follow HF revision
2bc89ac599031fa673cab993f1df02fc4a98c673, inference/image_processor.py.
"""

import logging
import math
import time
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.metrics import GaugeMetrics
from rtp_llm.multimodal.vit_metrics import (
    record_vit_preprocess_value,
    vit_preprocess_timer,
)

register_heif_opener()
_METRIC_TAGS = {"model": "deepseek_v41", "mm_type": "image"}


def _image_download_chunks(response, deadline, cancellation_event):
    from rtp_llm.multimodal.multimodal_util import _check_download

    raw = getattr(response, "raw", None)
    read1 = getattr(raw, "read1", None)
    if callable(read1):
        while True:
            _check_download(deadline, cancellation_event)
            connection = getattr(raw, "connection", None)
            sock = getattr(connection, "sock", None)
            if sock is not None:
                sock.settimeout(min(20, max(0.001, deadline - time.monotonic())))
            # Do not wait for a full chunk on a slow-drip response.
            chunk = read1(64 * 1024, decode_content=True)
            if not chunk:
                break
            yield chunk
    else:
        yield from response.iter_content(chunk_size=1)


def load_image_source(
    url,
    download_headers="",
    max_file_size_kb=10 * 1024,
    *,
    deadline=None,
    cancellation_event=None,
):
    """Load original bytes in the backend, retaining the V4.1 HTTP contract.

    HTTP retries share one deadline and never cache a mutable URL across
    requests. Local/data inputs use the standard loader. No image conversion
    occurs here; content inspection and ViT must consume the same source.
    """
    from rtp_llm.multimodal.multimodal_util import (
        _check_download,
        _get_http_heads,
        _record_download_time,
        get_bytes_io_from_url,
        request_get,
    )

    max_bytes = 10 * 1024 * 1024
    if max_file_size_kb is not None and max_file_size_kb > 0:
        max_bytes = min(max_bytes, max_file_size_kb * 1024)
    deadline = (
        min(deadline, time.monotonic() + 60)
        if deadline is not None
        else time.monotonic() + 60
    )
    if not url.startswith(("http://", "https://")):
        return get_bytes_io_from_url(
            url,
            download_headers,
            max_file_size_kb=max_bytes // 1024,
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
    for attempt in range(3):
        started = time.monotonic()
        status = None
        download_recorded = False
        try:
            _check_download(deadline, cancellation_event)
            remaining = deadline - time.monotonic()
            with request_get(
                url,
                _get_http_heads(download_headers),
                timeout=(min(5, remaining), min(20, remaining)),
            ) as response:
                status = response.status_code
                if status in (408, 429, 500, 502, 503, 504):
                    raise requests.ConnectionError("temporary image download failure")
                if status != 200:
                    raise ValueError("image download failed")
                content = BytesIO()
                for chunk in _image_download_chunks(
                    response, deadline, cancellation_event
                ):
                    _check_download(deadline, cancellation_event)
                    if content.tell() + len(chunk) > max_bytes:
                        raise ValueError("Multimodal file size is too large")
                    content.write(chunk)
                _check_download(deadline, cancellation_event)
                content.seek(0)
                return content
        except (
            requests.RequestException,
            ReadTimeoutError,
            ProtocolError,
            TimeoutError,
            ValueError,
        ) as error:
            # Network timing excludes retry backoff and is recorded exactly once.
            _record_download_time(started)
            download_recorded = True
            # Never log a signed URL, response body, or exception text.
            logging.warning(
                "V4.1 image download host=%s attempt=%d status=%s error_type=%s elapsed_ms=%.1f",
                urlsplit(url).hostname,
                attempt + 1,
                status,
                type(error).__name__,
                (time.monotonic() - started) * 1000,
            )
            transient = isinstance(
                error,
                (
                    requests.Timeout,
                    requests.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    ReadTimeoutError,
                    ProtocolError,
                    TimeoutError,
                ),
            )
            if not transient:
                message = (
                    "Multimodal file size is too large"
                    if isinstance(error, ValueError)
                    and str(error) == "Multimodal file size is too large"
                    else "Failed to download multimodal content"
                )
                raise FtRuntimeException(
                    ExceptionType.INVALID_PARAMS, message
                ) from error
            remaining = deadline - time.monotonic()
            if attempt == 2 or remaining <= 0:
                code = (
                    ExceptionType.CONNECT_TIMEOUT
                    if isinstance(
                        error, (requests.Timeout, ReadTimeoutError, TimeoutError)
                    )
                    else ExceptionType.MM_DOWNLOAD_FAILED
                )
                raise FtRuntimeException(
                    code, "Temporary multimodal download failure"
                ) from error
            delay = min(0.2 * 2**attempt, remaining)
            if cancellation_event is not None:
                cancellation_event.wait(delay)
                _check_download(deadline, cancellation_event)
            else:
                time.sleep(delay)
        finally:
            if not download_recorded:
                _record_download_time(started)


@dataclass(frozen=True)
class V41ImageProcessorConfig:
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936

    @classmethod
    def from_model_config(cls, config):
        vision = config.vision
        return cls(
            vision_patch_size=vision["patch_size"],
            vision_downsample_ratio=vision["downsample_ratio"],
            vision_max_n_token=vision["max_image_tokens"],
            vision_min_pixels=vision["min_pixels"],
        )


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
    # Pillow decodes lazily; conversion must be inside the decode timer.
    with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, _METRIC_TAGS):
        image = image.convert("RGB")
    return _preprocess_rgb_image(image, config)


def _preprocess_rgb_image(image: Image.Image, config: V41ImageProcessorConfig):
    with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, _METRIC_TAGS):
        llm_h, llm_w, best_height, best_width = plan_image_grid(
            image.width, image.height, config
        )
        patch = config.vision_patch_size
        vit_h, vit_w = best_height // patch, best_width // patch
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    record_vit_preprocess_value(
        GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC,
        best_height * best_width,
        _METRIC_TAGS,
    )
    with vit_preprocess_timer(
        GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC, _METRIC_TAGS
    ):
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


def _rgb_tensor(image: Image.Image) -> torch.Tensor:
    """Decode/convert on CPU and own the RGB bytes before closing the image."""
    with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, _METRIC_TAGS):
        return torch.from_numpy(
            np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        )


def _preprocess_rgb_on_device(rgb, target_hw, patch_size, device):
    """Only the GPU-owning caller may enter this boundary; no D2H round trip."""
    from rtp_llm.models_py.modules.dsv41.vision_preprocess_triton import (
        is_supported_shape,
        preprocess_rgb_u8,
    )
    from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("V4.1 GPU preprocessing requires an explicit CUDA device")
    if not is_supported_shape(tuple(rgb.shape[:2]), target_hw, patch_size=patch_size):
        raise ValueError("unsupported shape/Pillow version for V4.1 GPU preprocessing")
    # Coordinate with any concurrent ViT graph capture. The low-level processor
    # is eager and performs both uint8 resize passes before BF16 patch folding.
    with torch.cuda.device(device), cuda_graph_gate.operation():
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("V4.1 GPU preprocessing cannot run inside CUDA capture")
        pixels = rgb.to(device=device)
        return preprocess_rgb_u8(pixels, target_hw, patch_size=patch_size)
