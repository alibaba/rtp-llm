"""Native compact V4.1 encoding and scatter into full local cache pages.

The byte ABI is the compact reader's row-interleaved 528/288/68 format. This is
an explicit kernel candidate behind DSV41_NATIVE_COMPACT_WRITER=1. CP SWA byte
sharding and PD protocol integration are caller responsibilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from rtp_llm.models_py.modules.dsv41._compact_writer_triton import encode_compact_kernel
from rtp_llm.models_py.modules.dsv41.cache_layout import ENCODINGS, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    _separate_outputs,
    is_supported,
)

MAX_ENCODE_BYTES = 64 * 1024 * 1024
_FORMAT_IDS = {CacheRegion.SWA: 0, CacheRegion.GLOBAL: 1, CacheRegion.INDEX_K: 2}


@dataclass(frozen=True)
class CompactWriteResult:
    output: torch.Tensor
    status: torch.Tensor

    def check(self) -> None:
        with torch.cuda.device(self.status.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "writer status must be checked after graph execution"
                )
        errors = self.status.detach().cpu()
        if torch.any(errors != 0):
            codes = sorted(set(errors.tolist()) - {0})
            raise RuntimeError(
                f"compact writer rejected rows: status={codes}; "
                "1=nonfinite input, 2=invalid destination, 4=nonfinite/zero scale"
            )


def _validate(values: torch.Tensor, region: CacheRegion) -> None:
    if os.environ.get("DSV41_NATIVE_COMPACT_WRITER", "0") != "1":
        raise RuntimeError(
            "native compact writer requires DSV41_NATIVE_COMPACT_WRITER=1"
        )
    if not is_supported(values):
        raise RuntimeError("native compact writer requires a Blackwell CUDA device")
    if region not in ENCODINGS:
        raise ValueError("unknown compact cache region")
    if (
        values.ndim != 2
        or values.dtype != torch.bfloat16
        or values.shape[1] != ENCODINGS[region].head_dim
        or not values.is_contiguous()
    ):
        raise ValueError(
            "compact quantization requires contiguous BF16 [rows, region head_dim]"
        )


def _status_buffer(
    values: torch.Tensor, status: Optional[torch.Tensor]
) -> torch.Tensor:
    if status is None:
        return torch.empty((values.shape[0],), dtype=torch.int32, device=values.device)
    if (
        status.dtype != torch.int32
        or tuple(status.shape) != (values.shape[0],)
        or status.device != values.device
        or not status.is_contiguous()
    ):
        raise ValueError("writer status must be contiguous CUDA int32 [rows]")
    return status


def encode_compact(
    values: torch.Tensor,
    region: CacheRegion,
    *,
    output: Optional[torch.Tensor] = None,
    status: Optional[torch.Tensor] = None,
) -> CompactWriteResult:
    """Return packed uint8 rows; payload and scale bytes match the official codecs.

    Invalid numeric rows produce zero bytes and nonzero status. The input is not
    modified. Pass stable output/status buffers when capturing a CUDA graph.
    """
    _validate(values, region)
    encoding = ENCODINGS[region]
    shape = (values.shape[0], encoding.entry_bytes)
    if shape[0] * shape[1] > MAX_ENCODE_BYTES:
        raise ValueError("compact encoding exceeds 64 MiB workspace; tile the rows")
    if output is None:
        output = torch.empty(shape, dtype=torch.uint8, device=values.device)
    if (
        tuple(output.shape) != shape
        or output.dtype != torch.uint8
        or output.device != values.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "encoded output must be contiguous CUDA uint8 [rows, row_bytes]"
        )
    status = _status_buffer(values, status)
    _separate_outputs((output, status), (values,))
    if values.shape[0]:
        encode_compact_kernel[(values.shape[0],)](
            values,
            status,
            output,
            status,
            DIM=encoding.head_dim,
            GROUP=encoding.group_size,
            ROW_BYTES=encoding.entry_bytes,
            FORMAT=_FORMAT_IDS[region],
            PAGED=False,
            PAGE_STRIDE=0,
            ENTRIES=1,
            NUM_PAGES=0,
            num_warps=4,
        )
    return CompactWriteResult(output, status)


def write_compact(
    values: torch.Tensor,
    pages: CompactPages,
    slot_mapping: torch.Tensor,
    *,
    status: Optional[torch.Tensor] = None,
) -> CompactWriteResult:
    """Encode and store each row at page*entries+offset, with -1 for padding.

    Non-padding destinations must be unique within a launch, as with RTP's other
    KV scatter operators. Page zero is reserved. A numeric error or invalid slot
    leaves that entire destination row unchanged and sets an explicit status.
    Page padding and rows not selected by slot_mapping are never written.
    """
    if not isinstance(pages, CompactPages):
        raise TypeError("native writer requires row-interleaved compact pages")
    _validate(values, pages.region)
    pages.validate(values.device)
    if (
        slot_mapping.device != values.device
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or tuple(slot_mapping.shape) != (values.shape[0],)
        or not slot_mapping.is_contiguous()
    ):
        raise ValueError("slot mapping must be contiguous CUDA int32/int64 [rows]")
    encoding = ENCODINGS[pages.region]
    status = _status_buffer(values, status)
    _separate_outputs((pages.data, status), (values, slot_mapping))
    if values.shape[0]:
        encode_compact_kernel[(values.shape[0],)](
            values,
            slot_mapping,
            pages.data,
            status,
            DIM=encoding.head_dim,
            GROUP=encoding.group_size,
            ROW_BYTES=encoding.entry_bytes,
            FORMAT=_FORMAT_IDS[pages.region],
            PAGED=True,
            PAGE_STRIDE=pages.data.stride(0),
            ENTRIES=pages.entries_per_page,
            NUM_PAGES=pages.data.shape[0],
            num_warps=4,
        )
    return CompactWriteResult(pages.data, status)
