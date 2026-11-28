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
from PIL import Image
from pillow_heif import register_heif_opener
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from rtp_llm.config.dsv41_config import v41_image_detail_size
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.metrics import GaugeMetrics
from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import resize_geometry
from rtp_llm.multimodal.vit_metrics import (
    record_vit_preprocess_value,
    vit_preprocess_timer,
)

import base64
import contextlib
import hashlib
import io
import json
from dataclasses import asdict, field, replace
from typing import Any, Callable, Mapping, Sequence
from rtp_llm.metrics import kmonitor
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.vit_metrics import collect_vit_preprocess_metrics

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

register_heif_opener()
_METRIC_TAGS = {"model": "deepseek_v41", "mm_type": "image"}
_IMAGE_CONNECT_TIMEOUT_S = 3
_IMAGE_READ_TIMEOUT_S = 20
_IMAGE_DOWNLOAD_BUDGET_S = 30


def image_download_deadline(deadline: float | None = None) -> float:
    local_deadline = time.monotonic() + _IMAGE_DOWNLOAD_BUDGET_S
    return local_deadline if deadline is None else min(deadline, local_deadline)


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
                sock.settimeout(
                    min(_IMAGE_READ_TIMEOUT_S, max(0.001, deadline - time.monotonic()))
                )
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

    HTTP inputs reuse cached source bytes by URL; on a miss, retries share
    one deadline. Local/data inputs use the standard loader. No image conversion
    occurs here; content inspection and ViT must consume the same source.
    """
    from rtp_llm.multimodal.multimodal_util import (
        _check_download,
        _get_http_heads,
        _record_download_time,
        get_bytes_io_from_url,
        request_get,
        url_data_cache_,
    )

    max_bytes = 10 * 1024 * 1024
    if max_file_size_kb is not None and max_file_size_kb > 0:
        max_bytes = min(max_bytes, max_file_size_kb * 1024)
    deadline = image_download_deadline(deadline)
    if not url.startswith(("http://", "https://")):
        return get_bytes_io_from_url(
            url,
            download_headers,
            max_file_size_kb=max_bytes // 1024,
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
    cached = url_data_cache_.check_cache(url)
    if cached is not None:
        try:
            _check_download(deadline, cancellation_event)
        except TimeoutError as error:
            raise FtRuntimeException(
                ExceptionType.MM_DOWNLOAD_FAILED,
                "Temporary multimodal download failure",
            ) from error
        source = cached.getvalue()
        if len(source) > max_bytes:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, "Multimodal file size is too large"
            )
        return BytesIO(source)

    for attempt in range(3):
        started = time.monotonic()
        status = None
        download_recorded = False
        try:
            _check_download(deadline, cancellation_event)
            with request_get(
                url,
                _get_http_heads(download_headers),
                timeout=(_IMAGE_CONNECT_TIMEOUT_S, _IMAGE_READ_TIMEOUT_S),
                deadline=deadline,
                cancellation_event=cancellation_event,
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
            # A caller may close or mutate its stream; retain a private cache buffer.
            url_data_cache_.insert_cache(url, content)
            return BytesIO(content.getvalue())
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
                raise FtRuntimeException(
                    ExceptionType.MM_DOWNLOAD_FAILED,
                    "Temporary multimodal download failure",
                ) from error
            delay = min(0.2 * 2**attempt, remaining)
            if cancellation_event is not None:
                cancellation_event.wait(delay)
            else:
                time.sleep(delay)
            # The next attempt checks cancellation/deadline inside the try, so
            # backoff deadline expiry keeps the same download error contract.
        finally:
            if not download_recorded:
                _record_download_time(started)


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


def _preprocess_rgb_image(
    image: Image.Image, config: V41ImageProcessorConfig, *, grid=None
):
    with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, _METRIC_TAGS):
        llm_h, llm_w, best_height, best_width = grid or plan_image_grid(
            image.width, image.height, config
        )
        patch = config.vision_patch_size
        vit_h, vit_w = best_height // patch, best_width // patch
        resize_h, resize_w, top, left = resize_geometry(
            image.height, image.width, best_height, best_width
        )
        image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)
        if image.size != (best_width, best_height):
            canvas = Image.new("RGB", (best_width, best_height), (127, 127, 127))
            canvas.paste(image, (left, top))
            image = canvas
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


def _preprocess_rgb_on_device(rgb, target_hw, patch_size, device, *, tables=None):
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
        raise ValueError("unsupported shape for V4.1 GPU preprocessing")
    # Coordinate with any concurrent ViT graph capture. The low-level processor
    # is eager and performs both uint8 resize passes before BF16 patch folding.
    with torch.cuda.device(device), cuda_graph_gate.operation():
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("V4.1 GPU preprocessing cannot run inside CUDA capture")
        pixels = rgb.to(device=device)
        return preprocess_rgb_u8(
            pixels, target_hw, patch_size=patch_size, tables=tables
        )


@contextlib.contextmanager
def _report_preprocess_metrics():
    from rtp_llm.multimodal.multimodal_util import collect_download_timing

    start_ns = time.monotonic_ns()
    with collect_vit_preprocess_metrics() as metrics, collect_download_timing() as download:
        try:
            yield
        finally:
            total_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
            download_ms = max(0.0, min(download.elapsed_ms, total_ms))
            # This path runs in the frontend; retain the reporter's actual role tags.
            try:
                for sample in metrics.samples:
                    kmonitor.report(sample.metric, sample.value, sample.tags)
                kmonitor.report(
                    GaugeMetrics.VIT_PREPROCESS_RT_METRIC, total_ms, _METRIC_TAGS
                )
                kmonitor.report(
                    GaugeMetrics.VIT_DOWNLOAD_RT_METRIC, download_ms, _METRIC_TAGS
                )
                kmonitor.report(
                    GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC,
                    max(0.0, total_ms - download_ms),
                    _METRIC_TAGS,
                )
            except Exception:
                logging.exception(
                    "Failed to report DeepSeek-V4.1 preprocessing metrics"
                )


@dataclass(frozen=True)
class V41ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor
    content_sha256: str
    processor_identity: str
    source_bytes: bytes = field(default=b"", repr=False)

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
    from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url

    return get_bytes_io_from_url(url).getvalue()


def prepare_vl_inputs(
    prompt: str,
    images: Sequence[Mapping[str, Any]],
    tokenizer,
    config: V41ImageProcessorConfig,
    *,
    url_loader=None,
    output_budget: int = 0,
    retain_source_bytes: bool = False,
) -> V41PreparedInputs:
    # output_budget is retained for callers; the generation layer applies limits.
    return prepare_vl_inputs_from_token_ids(
        tokenizer.encode(prompt),
        images,
        config,
        prompt=prompt,
        url_loader=url_loader,
        retain_source_bytes=retain_source_bytes,
    )


def prepare_vl_inputs_from_token_ids(
    prompt_tokens: Sequence[int],
    images: Sequence[Mapping[str, Any]],
    config: V41ImageProcessorConfig,
    *,
    prompt: str = "",
    url_loader=None,
    max_image_bytes: int | None = None,
    retain_source_bytes: bool = False,
) -> V41PreparedInputs:
    """Expand upstream image markers without re-tokenizing any text or special IDs."""
    if sum(token == config.image_token_id for token in prompt_tokens) != len(images):
        raise ValueError(
            "image placeholder count does not match the supplied image records"
        )
    metrics_context = (
        _report_preprocess_metrics() if len(images) else contextlib.nullcontext()
    )
    with metrics_context:
        tokens, token_types, image_inputs = [], [], []
        records = iter(images)
        for token in prompt_tokens:
            if token != config.image_token_id:
                tokens.append(token)
                token_types.append(TEXT)
                continue
            record = next(records)
            detail_size = v41_image_detail_size(record.get("detail"))
            with vit_preprocess_timer(
                GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, _METRIC_TAGS
            ):
                data = load_image_bytes(record, url_loader=url_loader)
            if max_image_bytes is not None and len(data) > max_image_bytes:
                raise ValueError("Multimodal file size is too large")
            try:
                with vit_preprocess_timer(
                    GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, _METRIC_TAGS
                ):
                    with Image.open(io.BytesIO(data)) as source:
                        image = source.convert("RGB")
                grid = plan_image_grid(
                    *(detail_size if min(detail_size) > 0 else image.size), config
                )
                patches, vit_h, vit_w, llm_h, llm_w = _preprocess_rgb_image(
                    image, config, grid=grid
                )
            except (OSError, Image.DecompressionBombError) as error:
                raise ValueError(MMErr.IMG_OPEN) from error
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
                    source_bytes=data if retain_source_bytes else b"",
                )
            )
            tokens.extend([config.image_token_id] * types.numel())
            token_types.extend(types.tolist())
        return V41PreparedInputs(
            prompt, tuple(tokens), tuple(token_types), tuple(image_inputs)
        )
