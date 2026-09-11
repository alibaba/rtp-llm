"""Checked DeepSelect TopK boundary for V4.1 indexer and logits candidates.

This is selection only. Probability normalization, TopP, random draws and
accept/reject sampling remain with the caller. The original input dtype and
selected values are preserved, including infinities; NaNs fail per row.
"""

import os
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41.native_aot import native_identity


def is_supported(values, k):
    return (
        values.is_cuda
        and torch.cuda.get_device_capability(values.device)[0] == 10
        and values.ndim == 2
        and values.dtype in (torch.bfloat16, torch.float32)
        and 0 < values.shape[1] < 2**23
        and type(k) is int
        and 1 <= k <= 4096
    )


def _aligned(rows, columns, dtype, device, alignment):
    elements = alignment // dtype.itemsize
    stride = (columns + elements - 1) // elements * elements
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :columns]


@dataclass(frozen=True)
class TopKSelection:
    values: torch.Tensor
    indices: torch.Tensor
    status: torch.Tensor

    def check(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("check TopK status only after Graph execution")
        if torch.any(self.status.detach().cpu() != 0):
            raise RuntimeError("DeepSelect rejected NaN, length or output metadata")


def topk(
    values, k, *, end=None, output_idx=None, sorted_index=False, sorted_value=False
):
    """Return exact selected values and safe int32 logical indices (-1 padding).

    Caller-owned output_idx keeps its allocation across Graph replays. Temporary
    alignment/sanitizing tensors belong to the active stream or captured private
    pool. No upstream value allocation or stale CPU length is used. Invalid rows
    are sanitized before calling the vendor, including its unchecked short-row
    path, and never return an addressable sentinel.
    """
    if os.environ.get("DSV41_DEEPSELECT") != "1" or not is_supported(values, k):
        raise RuntimeError(
            "DeepSelect requires explicit supported V4.1 CUDA TopK dispatch"
        )
    if sorted_value and (values.dtype != torch.float32 or sorted_index):
        raise ValueError("value sorting requires FP32 and excludes index sorting")
    native_identity("deep-select")
    import deep_select

    if tuple(deep_select.get_stride_requirement()) != (1024, 32):
        raise RuntimeError("DeepSelect alignment differs from the pinned interface")
    rows, columns = values.shape
    if end is None:
        end = torch.full((rows,), columns, dtype=torch.int32, device=values.device)
    if (
        end.shape != (rows,)
        or end.dtype != torch.int32
        or end.device != values.device
        or not end.is_contiguous()
    ):
        raise ValueError("TopK end must be contiguous int32 with one length per row")
    if output_idx is None:
        output_idx = _aligned(rows, k, torch.int32, values.device, 32)
    padded_k = (k + 7) // 8 * 8
    if (
        output_idx.shape != (rows, k)
        or output_idx.dtype != torch.int32
        or output_idx.device != values.device
        or output_idx.stride(1) != 1
        or output_idx.data_ptr() % 32
        or output_idx.stride(0) < padded_k
        or output_idx.stride(0) * output_idx.element_size() % 32
    ):
        raise ValueError(
            "TopK output must have nonoverlapping int32 rows aligned to 32 bytes"
        )
    if rows and (
        (output_idx.storage_offset() + (rows - 1) * output_idx.stride(0) + padded_k) * 4
        > output_idx.untyped_storage().nbytes()
        or output_idx.untyped_storage().data_ptr()
        in (values.untyped_storage().data_ptr(), end.untyped_storage().data_ptr())
    ):
        raise ValueError("TopK output needs padded storage and must not alias inputs")
    if rows == 0:
        return TopKSelection(values.new_empty((0, k)), output_idx, end.clone())
    bad_lengths = (end < 0) | (end > columns)
    valid = torch.arange(columns, device=values.device)[None, :] < end[:, None]
    invalid = bad_lengths | (torch.isnan(values) & valid).any(-1)
    scratch = _aligned(rows, columns, values.dtype, values.device, 1024)
    scratch.copy_(torch.where(valid & ~invalid[:, None], values, -torch.inf))
    safe_end = torch.where(invalid, 0, end)
    deep_select.topk(
        scratch,
        k,
        end=safe_end,
        output_idx=output_idx,
        indices_type=torch.int32,
        sorted_index=sorted_index,
        idx_oob_fill_value=-1,
        return_value=False,
        abort_when_nan_found=False,
    )
    malformed = ((output_idx < -1) | (output_idx >= safe_end[:, None])).any(-1)
    malformed |= (output_idx >= 0).sum(-1) != safe_end.clamp(max=k)
    ordered = output_idx.sort(dim=-1).values
    malformed |= ((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])).any(-1)
    invalid |= malformed
    output_idx.masked_fill_(invalid[:, None], -1)
    selected = values.gather(1, output_idx.clamp_min(0).long())
    selected = selected.masked_fill(output_idx < 0, -torch.inf)
    if sorted_value:
        selected, order = selected.sort(dim=-1, descending=True, stable=True)
        output_idx.copy_(output_idx.gather(1, order))
    return TopKSelection(selected, output_idx, invalid.to(torch.int32))


def sampler_topk(logits, k, *, output_idx=None):
    """Candidate logits for the existing sampler; RNG and probabilities are untouched."""
    if logits.dtype != torch.float32:
        raise ValueError("V4.1 sampler TopK requires the official FP32 logits")
    return topk(logits, k, output_idx=output_idx, sorted_value=True)
