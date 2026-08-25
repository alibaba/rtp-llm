"""Shared Kimi K3 request-side multimodal prompt preparation."""

import asyncio
from typing import Callable, Sequence

import torch
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    K3_MAX_IMAGE_FILE_SIZE_KB,
    KimiK3VisionProcessor,
)

KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
_KIMI_K3_MEDIA_CONTENT = "<|media_content|>"
_MAX_IMAGES_PER_REQUEST = 16
_MAX_TOTAL_IMAGE_BYTES = 128 * 1024 * 1024


def _preflight_image(
    url: str, download_headers: str
) -> tuple[torch.Tensor, tuple[int, int]]:
    from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url

    data = get_bytes_io_from_url(
        url,
        download_headers,
        max_file_size_kb=K3_MAX_IMAGE_FILE_SIZE_KB,
    )
    with Image.open(data) as image:
        size = image.size
    return torch.frombuffer(data.getbuffer(), dtype=torch.uint8), size


async def prepare_kimi_k3_multimodal_prompt(
    prompt: str,
    urls: Sequence[str],
    encode: Callable[[str], list[int]],
    download_headers: str = "",
) -> tuple[list[int], list[torch.Tensor]]:
    """Expand K3 image placeholders and preflight each image exactly once."""
    if len(urls) > _MAX_IMAGES_PER_REQUEST:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 image count exceeds the per-request limit: "
            f"{len(urls)} > {_MAX_IMAGES_PER_REQUEST}",
        )

    placeholder_count = prompt.count(KIMI_K3_IMAGE_PLACEHOLDER)
    expanded_count = prompt.count(_KIMI_K3_MEDIA_CONTENT)
    if placeholder_count:
        if expanded_count or placeholder_count != len(urls):
            raise FtRuntimeException(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                "Kimi K3 image placeholder count does not match multimodal input "
                f"count: {placeholder_count} != {len(urls)}",
            )
    elif expanded_count != len(urls):
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 media prompt count does not match multimodal input count: "
            f"{expanded_count} != {len(urls)}",
        )

    preflighted = await asyncio.gather(
        *(
            asyncio.to_thread(_preflight_image, url, download_headers)
            for url in urls
        )
    )
    tensors = [tensor for tensor, _ in preflighted]
    total_bytes = sum(tensor.numel() for tensor in tensors)
    if total_bytes > _MAX_TOTAL_IMAGE_BYTES:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 image bytes exceed the per-request limit: "
            f"{total_bytes} > {_MAX_TOTAL_IMAGE_BYTES}",
        )

    expanded_prompt = prompt
    if placeholder_count:
        for _, (width, height) in preflighted:
            expanded_prompt = expanded_prompt.replace(
                KIMI_K3_IMAGE_PLACEHOLDER,
                KimiK3VisionProcessor.make_image_prompt(width, height),
                1,
            )

    token_ids = list(encode(expanded_prompt))
    if not token_ids:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 image prompt could not be tokenized",
        )
    return token_ids, tensors
