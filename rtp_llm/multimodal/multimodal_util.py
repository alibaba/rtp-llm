from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import re
import threading
from io import BytesIO
from typing import Any, List, Optional

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
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.lru_dict import LruDict

download_executor = concurrent.futures.ThreadPoolExecutor()

logger = logging.getLogger(__name__)

REQUEST_GET = None


def _default_request_get(url, headers):
    import requests

    return requests.get(url, stream=True, headers=headers, timeout=10)


def request_get(url, headers):
    global REQUEST_GET
    if REQUEST_GET is None:
        try:
            from internal_source.rtp_llm.utils.ssrf_check import safe_request_get

            REQUEST_GET = safe_request_get
        except ImportError:
            REQUEST_GET = _default_request_get
    return REQUEST_GET(url, headers)


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
    return res


def _validate_file_size(size_bytes: int, max_file_size_kb: Optional[int]) -> None:
    if max_file_size_kb is None or max_file_size_kb <= 0:
        return
    if size_bytes > max_file_size_kb * 1024:
        raise_mm(MMErr.FILE_TOO_LARGE)


def _download_http_content(
    url: str, headers: dict, max_file_size_kb: Optional[int]
) -> BytesIO:
    import requests

    response = None
    try:
        response = request_get(url, headers)
        if response.status_code != 200:
            raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)

        if max_file_size_kb is not None and max_file_size_kb > 0:
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise_mm(MMErr.MISS_CONTENT_LEN)
            try:
                content_length_bytes = int(content_length)
            except (TypeError, ValueError):
                raise_mm(MMErr.MISS_CONTENT_LEN)
            _validate_file_size(content_length_bytes, max_file_size_kb)

        content = BytesIO()
        downloaded_bytes = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded_bytes += len(chunk)
            _validate_file_size(downloaded_bytes, max_file_size_kb)
            content.write(chunk)
        content.seek(0)
        return content
    except FtRuntimeException:
        raise
    except (requests.exceptions.Timeout, TimeoutError):
        raise_mm(MMErr.DL_TIMEOUT, ExceptionType.MM_PROCESS_ERROR)
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
        requests.exceptions.InvalidSchema,
        requests.exceptions.URLRequired,
    ):
        raise_mm(MMErr.URL_INVALID, ExceptionType.MM_PROCESS_ERROR)
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
):
    """Get BytesIO from URL.

    Args:
        url: URL to fetch from.
        download_headers: JSON string containing HTTP headers. If empty, uses default headers.
        max_file_size_kb: Maximum file size in KB. HTTP URLs must provide a
            Content-Length header so the limit can be checked before reading the body.
    """

    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        headers = _get_http_heads(download_headers)
        try:
            if url.startswith("http") or url.startswith("https"):
                res = _download_http_content(url, headers, max_file_size_kb)
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
            _validate_file_size(res.getbuffer().nbytes, max_file_size_kb)
        except FtRuntimeException:
            raise
        except Exception:
            logger.exception("failed to load multimodal content")
            raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)
        url_data_cache_.insert_cache(url, res)
        return res
    else:
        cached_res.seek(0)
        _validate_file_size(cached_res.getbuffer().nbytes, max_file_size_kb)
        return cached_res


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


def build_multimodal_output_pb(
    embeddings: Optional[List[torch.Tensor]],
    position_ids: Optional[List[torch.Tensor]],
    extra_input: Optional[List[torch.Tensor]],
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
    if position_ids:
        output_pb.multimodal_pos_id.CopyFrom(
            trans_from_tensor(torch.concat(position_ids))
        )
    for extra in extra_input:
        output_pb.multimodal_extra_input.append(trans_from_tensor(extra))
    return output_pb
