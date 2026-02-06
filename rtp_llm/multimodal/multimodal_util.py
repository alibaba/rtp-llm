import base64
import concurrent.futures
import json
import logging
import re
import threading
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import Optional

import requests
import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMPreprocessConfigPB,
    MultimodalInputsPB,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.grpc_util import trans_tensor
from rtp_llm.utils.lru_dict import LruDict
from rtp_llm.utils.oss_util import get_bytes_io_from_oss_path

download_executor = concurrent.futures.ThreadPoolExecutor()

logger = logging.getLogger(__name__)

REQUEST_GET = None


def request_get(url, headers):
    global REQUEST_GET
    if REQUEST_GET is None:
        try:
            from internal_source.rtp_llm.utils.ssrf_check import safe_request_get

            REQUEST_GET = safe_request_get
        except ImportError:
            REQUEST_GET = lambda url, headers: requests.get(
                url, stream=True, headers=headers, timeout=10
            )
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


def get_bytes_io_from_url(url: str, download_headers: str = ""):
    """Get BytesIO from URL.

    Args:
        url: URL to fetch from.
        download_headers: JSON string containing HTTP headers. If empty, uses default headers.
    """

    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        headers = _get_http_heads(download_headers)
        try:
            if url.startswith("http") or url.startswith("https"):
                response = request_get(url, headers)
                if response.status_code == 200:
                    res = BytesIO(response.content)
                else:
                    raise Exception(
                        f"download failed, error code: {response.status_code}"
                    )
            elif url.startswith("oss"):
                res = get_bytes_io_from_oss_path(url)
            elif get_base64_prefix(url) > 0:
                res = BytesIO(base64.b64decode(url[get_base64_prefix(url) :]))
            else:
                # treat url as local path
                with open(url, "rb") as fh:
                    buf = BytesIO(fh.read())
                res = buf
        except Exception as e:
            raise Exception(f"download and load {url} error, exception {e}")
        url_data_cache_.insert_cache(url, res)
        return res
    else:
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
                mm_input.config,
            )
            for mm_input in multimodal_inputs
        ]
    else:
        raise ValueError(
            f"Unsupported multimodal input type: {type(multimodal_inputs)}"
        )
