from __future__ import annotations

import base64
import concurrent.futures
import contextvars
import json
import logging
import math
import re
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import Any, List, Optional, Sequence

import requests
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.lru_dict import LruDict

download_executor = concurrent.futures.ThreadPoolExecutor()

logger = logging.getLogger(__name__)


class _DownloadTiming:
    """Mutable timing carrier used while one preprocess call is running."""

    __slots__ = ("elapsed_ms",)

    def __init__(self):
        self.elapsed_ms = 0.0


_download_timing: contextvars.ContextVar[Optional[_DownloadTiming]] = (
    contextvars.ContextVar("multimodal_download_timing", default=None)
)


@contextmanager
def collect_download_timing():
    """Collect media-loading time for the enclosing preprocess call.

    The timer is context-local so concurrent preprocess calls cannot add their
    download durations to one another. A cache hit does not enter the timed
    load section and therefore contributes zero download time.
    """
    timing = _DownloadTiming()
    token = _download_timing.set(timing)
    try:
        yield timing
    finally:
        _download_timing.reset(token)


def _record_download_time(start_time: float) -> None:
    timing = _download_timing.get()
    if timing is not None:
        timing.elapsed_ms += max(0.0, time.monotonic() - start_time) * 1000.0


REQUEST_GET = None
CONNECT_TIMEOUT_RETRIES = 2


def _default_request_get(url, headers, timeout=10):
    import requests

    return requests.get(url, stream=True, headers=headers, timeout=timeout)


def _check_download(deadline, cancellation_event):
    if cancellation_event is not None and cancellation_event.is_set():
        raise concurrent.futures.CancelledError("Media download cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Media download deadline expired")


def request_get(url, headers, *, deadline=None, cancellation_event=None):
    global REQUEST_GET
    if REQUEST_GET is None:
        try:
            from internal_source.rtp_llm.utils.ssrf_check import safe_request_get

            REQUEST_GET = safe_request_get
        except ImportError:
            REQUEST_GET = _default_request_get

    import requests

    for retry_count in range(CONNECT_TIMEOUT_RETRIES + 1):
        _check_download(deadline, cancellation_event)
        try:
            if deadline is not None:
                return REQUEST_GET(
                    url,
                    headers,
                    timeout=max(0.001, min(10, deadline - time.monotonic())),
                )
            return REQUEST_GET(url, headers)
        except requests.exceptions.ConnectTimeout:
            if retry_count == CONNECT_TIMEOUT_RETRIES:
                raise
            logger.warning(
                "multimodal download connect timeout; retrying request (%d/%d)",
                retry_count + 1,
                CONNECT_TIMEOUT_RETRIES,
            )


def _get_http_heads(download_headers: str = ""):
    """Get HTTP headers from download_headers string.

    Args:
        download_headers: JSON string containing HTTP headers. If empty, returns default headers.
    """
    if download_headers != "":
        return json.loads(download_headers)
    else:
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        }


def get_base64_prefix(s):
    match = re.match(r"^data:[^,]*;base64,", s)
    if not match:
        return 0
    return match.end()


FRAMES_PACK_URL_PREFIX = "frames-pack:base64,"


def is_frames_pack_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(FRAMES_PACK_URL_PREFIX)


D3_VIDEO_GROUP_URL_PREFIX = "frames-pack:base64,"

# Envelope "version" of the d3 video-group contract. Bump on any field change.
# v3: the group's two frames ride losslessly inside the URL. The frontend RPC
# does not transport MultimodalInput.tensor (the python client never writes it
# into MultimodalInputPB and the C++ restore substitutes a 1-element
# placeholder, QueryConverter.cc), so the URL is the only channel that carries
# the frames across the RPC; the tensor stays as an in-process fast path.
D3_VIDEO_GROUP_VERSION = 3
D3_VIDEO_GROUP_MODE = "d3_video_group"
# Strict bounds the consumer re-validates (mirror of the frontend constants).
D3_VIDEO_GROUP_MAX_GROUPS = 16
D3_VIDEO_GROUP_FACTOR = 32
D3_VIDEO_GROUP_PATCH = 16
# The official producer's per-frame pixel ceiling: video_resize_dims caps every
# frame at VIDEO_TOKENS_PER_FRAME * IMAGE_FACTOR**2 = 196 * 32 * 32 = 200,704
# pixels (the checkpoint processor's cap_pixels_per_frame). The codec enforces
# the same ceiling on BOTH sides so a forged envelope can never advertise
# dimensions that imply an unbounded frames payload.
D3_VIDEO_GROUP_MAX_FRAME_PIXELS = 196 * 32 * 32
# video_key is the producer's sha256 content hash (64 hex chars). The bound
# keeps every identifying field — video_key, group_index, total_groups — inside
# the engine media cache key's first 256 encoded chars (url.substr(0, 256),
# MultimodalInputClass.h): a longer key could push group_index outside the
# truncated prefix and let different groups of one clip collide in the cache.
D3_VIDEO_GROUP_MAX_KEY_CHARS = 64


def is_d3_video_group_url(url: str) -> bool:
    """True for a d3 video-group envelope URL (one temporal group of one clip)."""
    if not isinstance(url, str) or not url.startswith(D3_VIDEO_GROUP_URL_PREFIX):
        return False
    try:
        envelope = _decode_d3_video_group_envelope(url)
    except ValueError:
        return False
    return envelope.get("mode") == D3_VIDEO_GROUP_MODE


def _decode_d3_video_group_envelope(url: str) -> dict:
    if not isinstance(url, str) or not url.startswith(D3_VIDEO_GROUP_URL_PREFIX):
        raise ValueError("not a d3 video-group url")
    body = url[len(D3_VIDEO_GROUP_URL_PREFIX) :]
    try:
        envelope = json.loads(base64.b64decode(body).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"d3 video-group envelope decode failed: {e}") from e
    if not isinstance(envelope, dict):
        raise ValueError("d3 video-group envelope must be a JSON object")
    return envelope


def _validate_d3_video_group_shape(height: int, width: int) -> None:
    """Frame-size bounds the codec enforces on encode AND parse.

    Multiples of 32 (already checked by callers) with the official per-frame
    pixel ceiling: the producer never emits larger frames, so anything beyond
    it is a forged envelope and must be rejected BEFORE the frames payload is
    sized, decoded or inflated.
    """
    if height * width > D3_VIDEO_GROUP_MAX_FRAME_PIXELS:
        raise ValueError(
            f"d3 video-group frame size {width}x{height} exceeds the official "
            f"per-frame ceiling of {D3_VIDEO_GROUP_MAX_FRAME_PIXELS} pixels"
        )


def encode_d3_video_group_url(
    video_key: str,
    group_index: int,
    total_groups: int,
    fps: float,
    frames_indices: Sequence[int],
    height: int,
    width: int,
    tokens: int,
    frames: bytes,
) -> str:
    """Encode one d3 video group (metadata + its two frames) as a URL.

    Each group is an INDEPENDENT unit: the envelope carries only this group's
    identity (which clip, which temporal group, the source fps and frame
    indices, the resized frame size, the token count) plus this group's two
    frames. ``frames`` are the group's raw uint8 RGB bytes in ``[2, 3, height,
    width]`` order, exactly as the producer resized them; they are embedded
    losslessly (zlib + base64) inside the URL because the RPC transports mm
    inputs by URL — ``MultimodalInput.tensor`` does not cross it. The frame
    size is bounded by the official per-frame ceiling before anything is
    compressed or serialized.
    """
    if not isinstance(video_key, str) or not video_key:
        raise ValueError("a d3 video group needs a nonempty video_key")
    if len(video_key) > D3_VIDEO_GROUP_MAX_KEY_CHARS:
        raise ValueError(
            f"a d3 video group video_key is at most {D3_VIDEO_GROUP_MAX_KEY_CHARS} chars"
        )
    _validate_d3_video_group_shape(int(height), int(width))
    frames = bytes(frames)
    expected = 2 * 3 * int(height) * int(width)
    if len(frames) != expected:
        raise ValueError(
            f"a d3 video group needs {expected} raw frame bytes for "
            f"{width}x{height}, got {len(frames)}"
        )
    envelope = {
        "version": D3_VIDEO_GROUP_VERSION,
        "mode": D3_VIDEO_GROUP_MODE,
        "video_key": str(video_key),
        "group_index": int(group_index),
        "total_groups": int(total_groups),
        "fps": float(fps),
        "frames_indices": [int(i) for i in frames_indices],
        "height": int(height),
        "width": int(width),
        "tokens": int(tokens),
        # Lossless frame payload: zlib-then-base64 keeps the bytes exact while
        # keeping the URL (and so the RPC message and the engine media-cache
        # key material) bounded; frames_bytes pins the size before inflating.
        "frames_bytes": len(frames),
        "frames": base64.b64encode(zlib.compress(frames, 6)).decode("ascii"),
    }
    payload = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    return D3_VIDEO_GROUP_URL_PREFIX + base64.b64encode(payload).decode("ascii")


def parse_d3_video_group_url(url: str) -> dict:
    """Decode and validate a d3 video-group envelope URL; ValueError on any gap.

    Returns the envelope with normalized int/float fields and the group's two
    frames as raw uint8 bytes under ``frames``. Every bound is enforced BEFORE
    any payload is decoded or inflated: the frame size (multiples of 32, the
    official per-frame pixel ceiling, which caps the payload at ~1.2 MB), the
    video_key length and the base64 payload length (a compressed stream can
    never usefully exceed the raw frame size, so a longer one is rejected
    without decoding it). Inflation itself is bounded: a zlib decompressor with
    a hard output cap of the declared size + 1 byte, so a stream that expands
    beyond it, is truncated, or carries trailing bytes is rejected without ever
    allocating beyond the declared frames.
    """
    envelope = _decode_d3_video_group_envelope(url)
    if envelope.get("version") != D3_VIDEO_GROUP_VERSION:
        raise ValueError(
            f"d3 video-group envelope version {envelope.get('version')!r} "
            f"is not {D3_VIDEO_GROUP_VERSION}"
        )
    if envelope.get("mode") != D3_VIDEO_GROUP_MODE:
        raise ValueError(
            f"not a d3 video-group envelope: mode {envelope.get('mode')!r}"
        )
    video_key = envelope.get("video_key")
    group_index = envelope.get("group_index")
    total_groups = envelope.get("total_groups")
    frames_indices = envelope.get("frames_indices")
    height = envelope.get("height")
    width = envelope.get("width")
    tokens = envelope.get("tokens")
    fps = envelope.get("fps")
    if not isinstance(video_key, str) or not video_key:
        raise ValueError("d3 video-group envelope needs a nonempty video_key")
    if len(video_key) > D3_VIDEO_GROUP_MAX_KEY_CHARS:
        raise ValueError("d3 video-group envelope video_key is too long")
    if type(group_index) is not int or type(total_groups) is not int:
        raise ValueError(
            "d3 video-group envelope needs integer group_index/total_groups"
        )
    if (
        not 0 <= group_index < total_groups
        or not 1 <= total_groups <= D3_VIDEO_GROUP_MAX_GROUPS
    ):
        raise ValueError(
            f"d3 video-group group_index {group_index} of {total_groups} groups is out of range"
        )
    if not isinstance(frames_indices, list) or len(frames_indices) != 2:
        raise ValueError("d3 video-group envelope needs the group's two frames_indices")
    if any(type(i) is not int or i < 0 for i in frames_indices):
        raise ValueError("d3 video-group frames_indices must be nonnegative integers")
    if type(height) is not int or type(width) is not int:
        raise ValueError("d3 video-group envelope needs integer height/width")
    if (
        height <= 0
        or width <= 0
        or height % D3_VIDEO_GROUP_FACTOR
        or width % D3_VIDEO_GROUP_FACTOR
    ):
        raise ValueError(
            f"d3 video-group frame size {width}x{height} must be positive multiples of "
            f"{D3_VIDEO_GROUP_FACTOR}"
        )
    _validate_d3_video_group_shape(height, width)
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("d3 video-group envelope needs a positive token count")
    if tokens != (height // D3_VIDEO_GROUP_FACTOR) * (width // D3_VIDEO_GROUP_FACTOR):
        raise ValueError("d3 video-group token count disagrees with the frame size")
    if type(fps) not in (int, float) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("d3 video-group envelope needs a positive finite fps")
    frames_bytes = envelope.get("frames_bytes")
    frames_payload = envelope.get("frames")
    expected_bytes = 2 * 3 * height * width
    if type(frames_bytes) is not int or frames_bytes != expected_bytes:
        raise ValueError(
            "d3 video-group envelope frames_bytes must equal the raw size of "
            f"two {width}x{height} frames ({expected_bytes})"
        )
    if not isinstance(frames_payload, str):
        raise ValueError("d3 video-group envelope needs a frames payload")
    # Bound the ENCODED payload before decoding it: a compressed stream never
    # usefully exceeds the raw frame size (zlib adds well under 0.1% on
    # incompressible data), so anything beyond base64 of the raw size plus
    # slack is rejected without allocating anything.
    max_encoded = (expected_bytes * 4 // 3) + 4096
    if len(frames_payload) > max_encoded:
        raise ValueError(
            f"d3 video-group frames payload is {len(frames_payload)} chars, over the "
            f"{max_encoded} bound for {width}x{height} frames"
        )
    try:
        compressed = base64.b64decode(frames_payload, validate=True)
    except Exception as e:
        raise ValueError(f"d3 video-group frames payload decode failed: {e}") from e
    # Bounded inflation: the decompressor may emit at most expected+1 bytes, so
    # an expansion bomb, a truncated stream or trailing junk is rejected without
    # ever allocating beyond the declared frames.
    inflater = zlib.decompressobj()
    try:
        frames = inflater.decompress(compressed, expected_bytes + 1)
    except zlib.error as e:
        raise ValueError(f"d3 video-group frames payload decode failed: {e}") from e
    if inflater.unconsumed_tail:
        raise ValueError(
            "d3 video-group frames payload expands beyond the declared frame size"
        )
    if not inflater.eof or inflater.unused_data:
        raise ValueError(
            "d3 video-group frames payload is truncated or carries trailing bytes"
        )
    if len(frames) != frames_bytes:
        raise ValueError(
            f"d3 video-group frames payload is {len(frames)} bytes, expected {frames_bytes}"
        )
    envelope["frames"] = frames
    return envelope


def encode_frames_pack_url(
    frames_jpeg_bytes: list, sampled_fps: float = 0.0, extras: Optional[dict] = None
) -> str:
    """Pack a list of raw JPEG byte payloads into a ``frames-pack:base64,...`` URL.

    The output URL stays inside ``MultimodalInput.url`` (a single string) so the
    proto wire stays unchanged. Each frame is inlined as
    ``data:image/jpeg;base64,...`` inside a JSON envelope so callers can also
    parse the body opaquely if needed.

    The packed format intentionally uses a custom URI scheme rather than
    ``data:application/json;base64,...`` so the open-source ``load_video``
    path (which feeds the URL to decord) never accidentally tries to decode
    a JSON blob as MPEG.
    """
    frames_data_urls = [
        "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
        for b in frames_jpeg_bytes
    ]
    envelope = {
        "version": 1,
        "sampled_fps": float(sampled_fps),
        "frames": frames_data_urls,
    }
    if extras:
        envelope.update(extras)
    payload = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    return FRAMES_PACK_URL_PREFIX + base64.b64encode(payload).decode("ascii")


def parse_frames_pack_url(url: str):
    """Decode a ``frames-pack:base64,...`` URL into a list of PIL Images + metadata.

    Returns ``(images, envelope)`` where ``images`` is a ``List[PIL.Image.Image]``
    and ``envelope`` is the JSON dict (so callers can read ``sampled_fps`` etc.).

    Raises ``ValueError`` if the URL doesn't have the expected prefix or the
    payload is malformed. This is a pure helper — does NOT touch network or
    filesystem.
    """
    if not is_frames_pack_url(url):
        raise ValueError(f"not a frames-pack url: {url[:64]!r}...")
    body = url[len(FRAMES_PACK_URL_PREFIX) :]
    try:
        envelope_bytes = base64.b64decode(body)
        envelope = json.loads(envelope_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"frames-pack envelope decode failed: {e}") from e
    frames = envelope.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames-pack envelope missing non-empty 'frames' list")
    from PIL import Image  # local import keeps top-level deps unchanged

    images = []
    for i, frame_url in enumerate(frames):
        if not isinstance(frame_url, str):
            raise ValueError(f"frames-pack frame[{i}] is not a string")
        prefix_len = get_base64_prefix(frame_url)
        if prefix_len == 0:
            raise ValueError(
                f"frames-pack frame[{i}] missing 'data:image/jpeg;base64,' prefix"
            )
        try:
            raw = base64.b64decode(frame_url[prefix_len:])
            images.append(Image.open(BytesIO(raw)).convert("RGB"))
        except Exception as e:
            raise ValueError(f"frames-pack frame[{i}] decode failed: {e}") from e
    return images, envelope


class IgraphItemKeyCountMismatchError(Exception):

    def __init__(self, requested_count: int, received_count: int, message: str = None):
        self.requested_count = requested_count
        self.received_count = received_count
        super().__init__(
            message
            or f"item number from igraph response ({received_count}) diff with keys number from request({requested_count})"
        )


def retry_on_assertion_error(retries: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    AssertionError,
                    ValueError,
                    IgraphItemKeyCountMismatchError,
                ) as e:
                    logger.warning(
                        f"[retry_on_assertion_error] AssertionError on attempt {attempt}: {str(e)}"
                    )
                    if attempt == retries:
                        logger.error(
                            "[retry_on_assertion_error] Max retries reached, re-raising."
                        )
                        raise

        return wrapper

    return decorator


def get_json_result_from_url(url: str, download_headers: str = ""):
    """Get JSON result from URL.

    Args:
        url: URL to fetch from.
        download_headers: JSON string containing HTTP headers. If empty, uses default headers.
    """
    headers = _get_http_heads(download_headers)
    load_start = time.monotonic()
    try:
        if url.startswith("http") or url.startswith("https"):
            import requests

            response = requests.get(url, stream=True, headers=headers, timeout=10)
            if response.status_code == 200:
                res = response.content.decode("utf-8")
            else:
                raise Exception(f"download failed, error code: {response.status_code}")
        elif get_base64_prefix(url) > 0:
            bytes_data = base64.b64decode(url[get_base64_prefix(url) :])
            res = bytes_data.decode("utf-8")
        else:
            # treat url as local path
            with open(url, "r", encoding="utf-8") as fh:
                buf = fh.read()
            res = buf
    except Exception as e:
        raise Exception(f"download and load {url} error, exception {e}")
    finally:
        _record_download_time(load_start)
    return res


def _validate_file_size(size_bytes: int, max_file_size_kb: Optional[int]) -> None:
    if max_file_size_kb is None or max_file_size_kb <= 0:
        return
    if size_bytes > max_file_size_kb * 1024:
        raise_mm(MMErr.FILE_TOO_LARGE)


def _download_http_content(
    url: str,
    headers: dict,
    max_file_size_kb: Optional[int],
    *,
    deadline=None,
    cancellation_event=None,
) -> BytesIO:
    import requests

    response = None
    try:
        if deadline is None and cancellation_event is None:
            response = request_get(url, headers)
        else:
            response = request_get(
                url, headers, deadline=deadline, cancellation_event=cancellation_event
            )
        if response.status_code != 200:
            raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)

        if max_file_size_kb is not None and max_file_size_kb > 0:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    content_length_bytes = int(content_length)
                except (TypeError, ValueError):
                    content_length_bytes = None
                if content_length_bytes is not None and content_length_bytes >= 0:
                    _validate_file_size(content_length_bytes, max_file_size_kb)

        content = BytesIO()
        downloaded_bytes = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            _check_download(deadline, cancellation_event)
            if not chunk:
                continue
            downloaded_bytes += len(chunk)
            _validate_file_size(downloaded_bytes, max_file_size_kb)
            content.write(chunk)
        _check_download(deadline, cancellation_event)
        content.seek(0)
        return content
    except FtRuntimeException:
        raise
    except concurrent.futures.CancelledError:
        raise
    except (requests.exceptions.Timeout, TimeoutError):
        raise_mm(MMErr.DL_TIMEOUT, ExceptionType.MM_DOWNLOAD_FAILED)
    except requests.exceptions.ConnectionError:
        raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)
    except (
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
        requests.exceptions.InvalidSchema,
        requests.exceptions.URLRequired,
    ):
        raise_mm(MMErr.URL_INVALID, ExceptionType.MM_WRONG_FORMAT_ERROR)
    except Exception:
        logger.exception("failed to download multimodal content")
        raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)
    finally:
        if response is not None:
            response.close()


def get_bytes_io_from_url(
    url: str,
    download_headers: str = "",
    max_file_size_kb: Optional[int] = VitConfig.DEFAULT_MM_IMAGE_MAX_FILE_SIZE_KB,
    *,
    deadline=None,
    cancellation_event=None,
):
    """Get BytesIO from URL.

    Args:
        url: URL to fetch from.
        download_headers: JSON string containing HTTP headers. If empty, uses default headers.
        max_file_size_kb: Maximum file size in KB. Content-Length is checked when
            available, and the limit is also enforced while streaming the body.
    """

    _check_download(deadline, cancellation_event)
    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        headers = _get_http_heads(download_headers)
        load_start = time.monotonic()
        try:
            if url.startswith("http") or url.startswith("https"):
                if deadline is None and cancellation_event is None:
                    res = _download_http_content(url, headers, max_file_size_kb)
                else:
                    res = _download_http_content(
                        url,
                        headers,
                        max_file_size_kb,
                        deadline=deadline,
                        cancellation_event=cancellation_event,
                    )
            elif url.startswith("oss"):
                from rtp_llm.utils.oss_util import get_bytes_io_from_oss_path

                res = get_bytes_io_from_oss_path(url)
            elif get_base64_prefix(url) > 0:
                res = BytesIO(base64.b64decode(url[get_base64_prefix(url) :]))
            else:
                # treat url as local path
                with open(url, "rb") as fh:
                    buf = BytesIO(fh.read())
                res = buf
            _check_download(deadline, cancellation_event)
            _validate_file_size(res.getbuffer().nbytes, max_file_size_kb)
        except (FtRuntimeException, concurrent.futures.CancelledError, TimeoutError):
            raise
        except Exception:
            logger.exception("failed to load multimodal content")
            raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)
        finally:
            _record_download_time(load_start)
        url_data_cache_.insert_cache(url, res)
        return res
    else:
        # Each caller needs an independent cursor because cache hits may be read
        # concurrently by different preprocessing workers.
        cached_bytes = cached_res.getvalue()
        _validate_file_size(len(cached_bytes), max_file_size_kb)
        return BytesIO(cached_bytes)


class MMDataCache(object):
    def __init__(self, cache_size: int = 10):
        self.mm_data_cache: Optional[LruDict] = None
        self.cache_lock = threading.Lock()
        if cache_size > 0:
            self.mm_data_cache = LruDict(cache_size)

    def check_cache(self, url: str):
        with self.cache_lock:
            if self.mm_data_cache == None:
                return None
            if url in self.mm_data_cache:
                return self.mm_data_cache[url]
            else:
                return None

    def insert_cache(self, url: str, data):
        with self.cache_lock:
            if self.mm_data_cache == None:
                return
            self.mm_data_cache[url] = data

    def resize_cache(self, cache_size: int):
        with self.cache_lock:
            if cache_size <= 0:
                self.mm_data_cache = None
                return
            if self.mm_data_cache is None:
                self.mm_data_cache = LruDict(cache_size)
            else:
                self.mm_data_cache.set_size(cache_size)


# Global cache instance for VIT embeddings
vit_emb_cache_ = MMDataCache(cache_size=10)
url_data_cache_ = MMDataCache(cache_size=10)


def trans_config(mm_process_config_pb: MMPreprocessConfigPB):
    return MMPreprocessConfig(
        width=mm_process_config_pb.width,
        height=mm_process_config_pb.height,
        min_pixels=mm_process_config_pb.min_pixels,
        max_pixels=mm_process_config_pb.max_pixels,
        fps=mm_process_config_pb.fps,
        min_frames=mm_process_config_pb.min_frames,
        max_frames=mm_process_config_pb.max_frames,
        crop_positions=list(mm_process_config_pb.crop_positions),
        mm_timeout_ms=mm_process_config_pb.mm_timeout_ms,
        max_long_side_pixel=(
            mm_process_config_pb.max_long_side_pixel
            if mm_process_config_pb.max_long_side_pixel > 0
            else -1
        ),
    )


def trans_mm_input(multimodal_inputs):
    # vit sep
    if isinstance(multimodal_inputs, MultimodalInputsPB):
        return [
            MultimodalInput(
                mm_input.multimodal_url,
                MMUrlType(mm_input.multimodal_type),
                trans_tensor(mm_input.multimodal_tensor),
                trans_config(mm_input.mm_preprocess_config),
            )
            for mm_input in multimodal_inputs.multimodal_inputs
        ]
    # not sep
    elif isinstance(multimodal_inputs, list):
        return [
            MultimodalInput(
                mm_input.url,
                MMUrlType(mm_input.mm_type),
                mm_input.tensor,
                mm_input.mm_preprocess_config,
            )
            for mm_input in multimodal_inputs
        ]
    else:
        raise ValueError(
            f"Unsupported multimodal input type: {type(multimodal_inputs)}"
        )


def maybe_tensor_to_list(tensor: Any, ndim_threshold: int = 2) -> Any:
    """Split a stacked tensor into a per-image list, or wrap a single tensor.

    `ndim_threshold` is a number-of-dimensions threshold (NOT a dim to operate
    on): a tensor with more than `ndim_threshold` dims is treated as stacked and
    split along its leading dim; otherwise it is wrapped in a single-element list.
    Non-tensor input is returned unchanged (hence the `Any` return type); None
    becomes [].
    """
    if tensor is None:
        return []
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if len(tensor.shape) > ndim_threshold:
        return list(tensor)
    return [tensor]


@cuda_graph_gate.operation()
def build_multimodal_output_pb(
    embeddings: Optional[List[torch.Tensor]],
    position_ids: Optional[List[torch.Tensor]],
    extra_input: Optional[List[torch.Tensor]],
    feature_hashes: Optional[List[torch.Tensor]] = None,
) -> MultimodalOutputPB:
    """Serialize embedding tensors into a MultimodalOutputPB."""
    embeddings = embeddings or []
    position_ids = position_ids or []
    extra_input = extra_input or []
    if not embeddings:
        return MultimodalOutputPB()
    output_pb = MultimodalOutputPB(
        multimodal_embedding=trans_from_tensor(torch.concat(embeddings)),
        split_size=[e.shape[0] for e in embeddings],
    )
    add_multimodal_feature_hashes(output_pb, embeddings, feature_hashes)
    if position_ids:
        output_pb.multimodal_pos_id.CopyFrom(
            trans_from_tensor(torch.concat(position_ids))
        )
    for extra in extra_input:
        output_pb.multimodal_extra_input.append(trans_from_tensor(extra))
    return output_pb


def add_multimodal_feature_hashes(output_pb, embeddings, feature_hashes):
    if feature_hashes is None:
        return
    if len(feature_hashes) != len(embeddings) or any(
        h.device.type != "cpu"
        or h.dtype != torch.int32
        or h.ndim != 1
        or h.numel() != e.shape[0]
        for e, h in zip(embeddings, feature_hashes)
    ):
        raise ValueError("invalid multimodal feature hashes")
    if feature_hashes:
        output_pb.multimodal_feature_hash.CopyFrom(
            trans_from_tensor(torch.cat(feature_hashes))
        )
