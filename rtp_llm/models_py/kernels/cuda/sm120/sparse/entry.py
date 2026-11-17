"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



from dataclasses import dataclass
import math
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast
import torch
from flashinfer.utils import (
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    is_sm12x_supported,
)


@dataclass(frozen=True)
class _SparseMLASegment:
    """Internal SM120 sparse MLA KV segment."""

    indices: torch.Tensor
    lengths: Optional[torch.Tensor] = None
    kv_cache: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _NormalizedSparseMLASegment:
    kv_cache: Optional[torch.Tensor]
    indices: torch.Tensor
    lengths: Optional[torch.Tensor]


def _normalize_optional_mla_sink(
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    backend_name: str,
) -> Optional[torch.Tensor]:
    if sinks is None:
        return None
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError(
                f"{backend_name} expects sinks to be a single tensor or a "
                f"length-1 list/tuple; got len={len(sinks)}."
            )
        return sinks[0]
    return sinks


def _normalize_sparse_mla_indices_and_lens(
    *,
    indices: torch.Tensor,
    lens: Optional[torch.Tensor],
    batch_size: int,
    q_len_per_request: int,
    device: torch.device,
    name: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if indices.ndim == 3:
        sparse_topk = int(indices.shape[-1])
        expected_shape: Tuple[int, ...] = (
            batch_size,
            q_len_per_request,
            sparse_topk,
        )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Expected {name}.shape == {expected_shape}, got {tuple(indices.shape)}"
            )
        indices = indices.reshape(batch_size * q_len_per_request, -1)
    elif indices.ndim == 2:
        sparse_topk = int(indices.shape[-1])
        expected_shape = (batch_size * q_len_per_request, sparse_topk)
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Expected flattened {name}.shape == {expected_shape}, got "
                f"{tuple(indices.shape)}"
            )
    else:
        raise ValueError(f"{name} must have ndim 2 or 3, got {indices.ndim}")
    if sparse_topk <= 0:
        raise ValueError(f"{name} requires top-k > 0")
    if indices.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {indices.dtype}")
    if indices.device != device:
        raise ValueError(f"{name} must be on device {device}, got {indices.device}")

    if lens is None:
        return indices, None
    if lens.dtype != torch.int32:
        raise ValueError(f"{name}_lens must have dtype torch.int32, got {lens.dtype}")
    if lens.device != device:
        raise ValueError(f"{name}_lens must be on device {device}, got {lens.device}")
    if lens.ndim == 2:
        expected_lens_shape: Tuple[int, ...] = (batch_size, q_len_per_request)
        if tuple(lens.shape) != expected_lens_shape:
            raise ValueError(
                f"Expected {name}_lens.shape == {expected_lens_shape}, got "
                f"{tuple(lens.shape)}"
            )
        lens = lens.reshape(-1)
    elif lens.ndim == 1:
        expected_lens_shape = (batch_size * q_len_per_request,)
        if tuple(lens.shape) != expected_lens_shape:
            raise ValueError(
                f"Expected flattened {name}_lens.shape == {expected_lens_shape}, "
                f"got {tuple(lens.shape)}"
            )
    else:
        raise ValueError(f"{name}_lens must have ndim 1 or 2, got {lens.ndim}")
    return indices, lens


def _normalize_sparse_mla_segments(
    *,
    sparse_mla_segments: Sequence[_SparseMLASegment],
    batch_size: int,
    q_len_per_request: int,
    device: torch.device,
) -> List[_NormalizedSparseMLASegment]:
    if not sparse_mla_segments:
        raise ValueError("backend='sparse' requires at least one sparse MLA segment")
    if len(sparse_mla_segments) > 2:
        raise NotImplementedError(
            "backend='sparse' currently supports at most two sparse MLA segments"
        )

    normalized: List[_NormalizedSparseMLASegment] = []
    for i, segment in enumerate(sparse_mla_segments):
        if not isinstance(segment, _SparseMLASegment):
            raise TypeError(
                "sparse_mla_segments must contain _SparseMLASegment instances; "
                f"got {type(segment)!r} at index {i}"
            )
        if i == 0 and segment.kv_cache is not None:
            raise ValueError(
                "sparse_mla_segments[0].kv_cache must be None; the first segment "
                "uses the function's kv_cache argument"
            )
        if i > 0:
            if segment.kv_cache is None:
                raise ValueError(
                    f"sparse_mla_segments[{i}].kv_cache is required for additional "
                    "sparse MLA segments"
                )
            if segment.kv_cache.device != device:
                raise ValueError(
                    f"sparse_mla_segments[{i}].kv_cache must be on device {device}, "
                    f"got {segment.kv_cache.device}"
                )

        indices, lengths = _normalize_sparse_mla_indices_and_lens(
            indices=segment.indices,
            lens=segment.lengths,
            batch_size=batch_size,
            q_len_per_request=q_len_per_request,
            device=device,
            name=f"sparse_mla_segments[{i}].indices",
        )
        normalized.append(
            _NormalizedSparseMLASegment(
                kv_cache=segment.kv_cache,
                indices=indices,
                lengths=lengths,
            )
        )
    return normalized


def _workspace_tensor_view(
    workspace_buffer: torch.Tensor,
    *,
    byte_offset: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], int]:
    if not workspace_buffer.is_contiguous():
        return None, byte_offset
    elem_size = torch.empty(
        (), dtype=dtype, device=workspace_buffer.device
    ).element_size()
    byte_offset = ((byte_offset + elem_size - 1) // elem_size) * elem_size
    numel = math.prod(shape)
    byte_end = byte_offset + numel * elem_size
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if byte_end > workspace_bytes:
        return None, byte_offset
    flat = workspace_buffer.view(torch.uint8)
    view = flat[byte_offset:byte_end].view(dtype).view(shape)
    return view, byte_end


def _sparse_mla_decode_workspace(
    workspace_buffer: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if num_tokens > 64:
        return None, None
    split_tile = 64
    num_splits = (topk + split_tile - 1) // split_tile + (
        extra_topk + split_tile - 1
    ) // split_tile
    mid_out, offset = _workspace_tensor_view(
        workspace_buffer,
        byte_offset=0,
        shape=(num_tokens, num_heads, num_splits, d_v),
        dtype=torch.bfloat16,
    )
    if mid_out is None:
        return None, None
    mid_lse, _ = _workspace_tensor_view(
        workspace_buffer,
        byte_offset=offset,
        shape=(num_tokens, num_heads, num_splits),
        dtype=torch.float32,
    )
    if mid_lse is None:
        return None, None
    return mid_out, mid_lse


def _trtllm_batch_decode_sparse_mla_sm120(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_mla_segments: Sequence[_SparseMLASegment],
    out: Optional[torch.Tensor],
    sm_scale: float,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    kv_scale_format: str,
    is_prefill: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not is_sm12x_supported(query.device):
        raise ValueError(
            "SM120 sparse MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 13.0)"
        )
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
    if kv_cache.device != query.device:
        raise ValueError(
            f"kv_cache must be on query device {query.device}, got {kv_cache.device}"
        )
    if workspace_buffer.device != query.device:
        raise ValueError(
            "workspace_buffer must be on query device "
            f"{query.device}, got {workspace_buffer.device}"
        )

    batch_size, q_len_per_request, num_heads, head_dim = query.shape
    if head_dim not in (512, 576):
        raise ValueError(
            "Sparse MLA supports DSv4 head_dim=512 or "
            f"DSv3.2/GLM head_dim=576, got {head_dim}"
        )
    if num_heads > 128:
        raise ValueError(f"Expected num_heads <= 128, got {num_heads}")

    segments = _normalize_sparse_mla_segments(
        sparse_mla_segments=sparse_mla_segments,
        batch_size=batch_size,
        q_len_per_request=q_len_per_request,
        device=query.device,
    )
    primary_segment = segments[0]
    extra_segment = segments[1] if len(segments) > 1 else None

    from ._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    query_flat = query.reshape(batch_size * q_len_per_request, num_heads, head_dim)
    expected_out_shape = (batch_size, q_len_per_request, num_heads, 512)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out, expected_out_shape, torch.bfloat16, query.device, "out"
        )
    out_flat = out.view(query_flat.shape[0], num_heads, 512)

    flat_lse_shape = (query_flat.shape[0], num_heads)
    nested_lse_shape = (batch_size, q_len_per_request, num_heads)
    user_lse = lse
    out_lse_arg = None
    if return_lse and lse is not None:
        if tuple(lse.shape) == flat_lse_shape:
            check_shape_dtype_device(
                lse, flat_lse_shape, torch.float32, query.device, "lse"
            )
        elif tuple(lse.shape) == nested_lse_shape:
            check_shape_dtype_device(
                lse, nested_lse_shape, torch.float32, query.device, "lse"
            )
            lse = lse.view(flat_lse_shape)
        else:
            raise ValueError(
                f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                f"got {tuple(lse.shape)}"
            )
        out_lse_arg = lse

    runner = _SparseMLAPagedAttentionRunner(
        max_num_tokens=query_flat.shape[0],
        max_num_heads=num_heads,
        kv_scale_format=kv_scale_format,
        device=query.device,
    )
    extra_topk = extra_segment.indices.shape[-1] if extra_segment is not None else 0
    mid_out, mid_lse = None, None
    if not is_prefill:
        mid_out, mid_lse = _sparse_mla_decode_workspace(
            workspace_buffer,
            num_tokens=query_flat.shape[0],
            num_heads=num_heads,
            d_v=512,
            topk=primary_segment.indices.shape[-1],
            extra_topk=extra_topk,
        )

    out_lse = runner.run(
        query_flat,
        kv_cache,
        primary_segment.indices,
        out_flat,
        float(sm_scale),
        topk_length=primary_segment.lengths,
        attn_sink=_normalize_optional_mla_sink(sinks, "backend='sparse'"),
        extra_kv_cache=extra_segment.kv_cache if extra_segment is not None else None,
        extra_indices=extra_segment.indices if extra_segment is not None else None,
        extra_topk_length=extra_segment.lengths if extra_segment is not None else None,
        out_lse=out_lse_arg,
        mid_out=mid_out,
        mid_lse=mid_lse,
        return_lse=return_lse,
        is_prefill=is_prefill,
    )

    if return_lse:
        return out, user_lse if user_lse is not None else out_lse
    return out


def _check_sm120_dsv4_kv_cache_layout(
    kv_cache: torch.Tensor,
    kv_layout: Literal["HND", "NHD"],
    name: str,
) -> torch.Tensor:
    if kv_cache.ndim == 3:
        return kv_cache
    if kv_cache.ndim != 4:
        raise ValueError(f"Expected {name}.ndim == 3 or 4, got {kv_cache.ndim}")
    if kv_layout == "HND":
        if kv_cache.size(1) != 1:
            raise ValueError(
                f"Expected packed SM120 DSV4 HND {name}.shape[1] == 1, got "
                f"{tuple(kv_cache.shape)}"
            )
    elif kv_layout == "NHD":
        if kv_cache.size(2) != 1:
            raise ValueError(
                f"Expected packed SM120 DSV4 NHD {name}.shape[2] == 1, got "
                f"{tuple(kv_cache.shape)}"
            )
    else:
        raise ValueError(f"kv_layout must be either 'HND' or 'NHD', got {kv_layout}")
    return kv_cache


def _trtllm_batch_decode_sparse_mla_dsv4_sm120(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor],
    swa_topk_lens: torch.Tensor,
    extra_sparse_indices: Optional[torch.Tensor],
    extra_sparse_topk_lens: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: Optional[torch.Tensor],
    kv_layout: Literal["HND", "NHD"],
    is_prefill: bool = False,
) -> torch.Tensor:
    if bmm2_scale != 1.0:
        raise ValueError("SM120 DSv4 sparse MLA does not support bmm2_scale")
    if query.ndim in (3, 4):
        num_heads, head_dim = query.shape[-2:]
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")
    if query.dtype != torch.bfloat16:
        raise ValueError(
            f"SM120 DSv4 sparse MLA only supports BF16 query, got {query.dtype}"
        )
    if head_dim != 512:
        raise ValueError(f"Expected DSv4 query head dim 512, got {head_dim}")
    if num_heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            "Expected 8, 16, 32, 64, or 128 query heads for SM120 DSv4 "
            f"sparse MLA, got {num_heads}"
        )
    if swa_topk_lens is None:
        raise ValueError("backend='sparse' requires swa_topk_lens")

    swa_kv_cache = _check_sm120_dsv4_kv_cache_layout(
        swa_kv_cache, kv_layout, "swa_kv_cache"
    )
    if swa_kv_cache.dtype == torch.uint8:
        if swa_kv_cache.size(-1) != 584:
            raise ValueError(
                "Expected packed SM120 DSV4 swa_kv_cache head dim 584, got "
                f"{swa_kv_cache.size(-1)}"
            )
    elif swa_kv_cache.dtype != query.dtype:
        raise ValueError(
            f"swa_kv_cache dtype must match query dtype, got {swa_kv_cache.dtype} "
            f"and {query.dtype}"
        )
    elif swa_kv_cache.size(-1) != 512:
        raise ValueError(
            f"Expected swa_kv_cache head dim 512, got {swa_kv_cache.size(-1)}"
        )

    if (extra_sparse_indices is None) != (extra_sparse_topk_lens is None):
        raise ValueError(
            "extra_sparse_indices and extra_sparse_topk_lens must be provided "
            "together for backend='sparse'"
        )
    query_for_sm120 = query if query.ndim == 4 else query.unsqueeze(1)
    out_for_sm120 = out if out is None or out.ndim == 4 else out.unsqueeze(1)

    sparse_mla_segments: List[_SparseMLASegment] = [
        _SparseMLASegment(indices=sparse_indices, lengths=swa_topk_lens)
    ]
    if extra_sparse_indices is not None:
        if compressed_kv_cache is None:
            raise ValueError(
                "compressed_kv_cache is required when extra_sparse_indices is provided"
            )
        compressed_kv_cache = _check_sm120_dsv4_kv_cache_layout(
            compressed_kv_cache, kv_layout, "compressed_kv_cache"
        )
        if compressed_kv_cache.dtype == torch.uint8:
            if compressed_kv_cache.size(-1) != 584:
                raise ValueError(
                    "Expected packed SM120 DSV4 compressed_kv_cache head dim 584, "
                    f"got {compressed_kv_cache.size(-1)}"
                )
        elif compressed_kv_cache.dtype != query.dtype:
            raise ValueError(
                "compressed_kv_cache dtype must match query dtype, got "
                f"{compressed_kv_cache.dtype} and {query.dtype}"
            )
        elif compressed_kv_cache.size(-1) != 512:
            raise ValueError(
                "Expected compressed_kv_cache head dim 512, got "
                f"{compressed_kv_cache.size(-1)}"
            )
        sparse_mla_segments.append(
            _SparseMLASegment(
                indices=extra_sparse_indices,
                lengths=extra_sparse_topk_lens,
                kv_cache=compressed_kv_cache,
            )
        )

    result = cast(
        torch.Tensor,
        _trtllm_batch_decode_sparse_mla_sm120(
            query=query_for_sm120,
            kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_mla_segments=sparse_mla_segments,
            out=out_for_sm120,
            sm_scale=bmm1_scale,
            sinks=sinks,
            lse=None,
            return_lse=False,
            kv_scale_format="auto",
            is_prefill=is_prefill,
        ),
    )
    if query.ndim == 3:
        return result.squeeze(1)
    return result


def trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor] = None,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[torch.Tensor] = None,
    kv_layout: Literal["HND", "NHD"] = "HND",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_pdl: bool | None = None,
    swa_topk_lens: Optional[torch.Tensor] = None,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    is_prefill: bool = False,
) -> torch.Tensor:
    """Run DeepSeek V4 packed sparse MLA prefill or decode on SM120.

    Query tensors are BF16 with head dimension 512. The required SWA pool
    and optional compressed pool are uint8 with 584 bytes per token. The
    pools share `kv_layout` and have independent index and top-k length
    tables. The operation writes BF16 output, using `out` when supplied.

    `is_prefill` selects the prefill kernel for short continuations that
    would otherwise dispatch by row count. Unused arguments preserve the
    upstream entrypoint signature.
    """
    cc = get_compute_capability(query.device)
    if cc != (12, 0):
        raise ValueError(f"RTP DSv4 sparse MLA requires SM120, got {cc}")
    backend = "sparse"
    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    if isinstance(bmm1_scale, torch.Tensor):
        if backend == "sparse":
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm1_scale to be a float"
            )
    if isinstance(bmm2_scale, torch.Tensor):
        if backend == "sparse":
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm2_scale to be a float"
            )
    if backend == "sparse":
        return _trtllm_batch_decode_sparse_mla_dsv4_sm120(
            query=query,
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            swa_topk_lens=swa_topk_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_topk_lens,
            out=out,
            bmm1_scale=float(bmm1_scale),
            bmm2_scale=float(bmm2_scale),
            sinks=sinks,
            kv_layout=kv_layout,
            is_prefill=is_prefill,
        )
