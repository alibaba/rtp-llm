from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

import torch

_sm120_sparse_mla = None
_sm120_sparse_mla_load_attempted = False


def _load_sm120_sparse_mla():
    """Load the SM120-only FlashInfer entry point on first SM120 use.

    Device selection can import the adapter without loading the optional
    CUDA wheel. Provider validation happens when SM120 execution is prepared.
    """
    global _sm120_sparse_mla, _sm120_sparse_mla_load_attempted
    if _sm120_sparse_mla_load_attempted:
        return _sm120_sparse_mla
    _sm120_sparse_mla_load_attempted = True
    try:
        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

        _sm120_sparse_mla = trtllm_batch_decode_sparse_mla_dsv4
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        logging.info("[dsv4-fp8] SM120 FlashInfer sparse MLA unavailable (%s)", error)
    return _sm120_sparse_mla


def prepare_decode(device: torch.device, n_heads: int) -> None:
    """Validate and warm the provider on the device selected for decode."""
    validate_sm120_head_count(n_heads)
    if _load_sm120_sparse_mla() is None:
        raise RuntimeError(
            "SM120 FP8 sparse decode requires FlashInfer "
            "trtllm_batch_decode_sparse_mla_dsv4; install the CUDA 13 "
            "RTP-LLM dependency set before starting the service"
        )
    warmup(device)


_WORKSPACES: dict[torch.device, torch.Tensor] = {}

# Head counts with a native FlashInfer SM120 sparse-MLA instantiation.
SM120_HEAD_COUNTS = (8, 16, 32, 64, 128)


def validate_sm120_head_count(heads: int) -> None:
    if heads not in SM120_HEAD_COUNTS:
        raise ValueError(
            "SM120 sparse MLA requires 8, 16, 32, 64 or 128 local query heads; "
            f"got {heads}. Reduce tensor parallelism or use a supported head count."
        )


@lru_cache(maxsize=4)
def _require_prefill_api(provider) -> None:
    try:
        supported = "is_prefill" in inspect.signature(provider).parameters
    except (TypeError, ValueError):
        supported = False
    if not supported:
        raise RuntimeError(
            "SM120 sparse prefill requires FlashInfer "
            "trtllm_batch_decode_sparse_mla_dsv4 with is_prefill support; "
            "install the qualified CUDA 13 FlashInfer dependency"
        )


# FlashInfer's DSV4 sparse-decode dispatcher specializes on the static index
# width.  HCA compresses one entry per 128 input tokens, so a 1M context needs
# an 8192-wide extra sparse table.  Keep the complete production width set in
# one place so eager and CUDA-graph preparation cannot diverge.
SM120_SWA_TOPK_WIDTHS = (128, 512, 1024)
SM120_EXTRA_TOPK_WIDTHS = (2, 128, 512, 1024, 2048, 4096, 8192)


def validate_sm120_swa_topk_width(
    requested_width: int,
    *,
    context: str,
) -> int:
    """Return the native-kernel width, or the requested generic-fallback width.

    FlashInfer precompiles the SM120 DSV4 decode launcher only for the widths
    in ``SM120_SWA_TOPK_WIDTHS``.  Canonicalization pads smaller requests to the
    next native instance.  Larger legal windows use the chunked generic path;
    rejecting them at model startup would incorrectly reject DSpARK's
    ``sliding_window + gamma`` width (for example 1152).
    """
    requested_width = int(requested_width)
    if requested_width <= 0:
        raise ValueError(
            f"{context} requires a positive SWA Top-K width, got {requested_width}"
        )
    kernel_width = next(
        (width for width in SM120_SWA_TOPK_WIDTHS if width >= requested_width),
        None,
    )
    return requested_width if kernel_width is None else kernel_width


def uses_generic_fallback(swa_width: int, extra_width: int = 0) -> bool:
    """Whether a sparse shape needs the graph-safe chunked implementation."""
    return int(swa_width) > SM120_SWA_TOPK_WIDTHS[-1] or int(extra_width) > (
        SM120_EXTRA_TOPK_WIDTHS[-1]
    )


def workspace(device: torch.device) -> torch.Tensor:
    result = _WORKSPACES.get(device)
    if result is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM120 sparse MLA workspace must be materialized before "
                "CUDA graph capture"
            )
        result = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = result
    return result


def warmup(device: torch.device) -> torch.Tensor:
    """Materialize the fixed FlashInfer workspace before graph capture."""
    return workspace(device)


def token_lens(
    lengths: Optional[torch.Tensor],
    rows: int,
    width: int,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize per-row sparse-index lengths for the FlashInfer ABI.

    ``lengths`` is a *prefix* length: the kernel scans the first ``length``
    entries of each row.  A few metadata producers can leave ``-1`` holes in
    that prefix (for example an empty page during the first decode step), so
    callers may provide ``valid_mask`` to derive the effective count instead.
    Keeping this conversion on the input device also avoids a host sync in
    CUDA-graph replay.
    """
    if rows < 0 or width < 0:
        raise ValueError(f"invalid sparse-index shape rows={rows}, width={width}")

    if lengths is None:
        if valid_mask is None:
            return torch.full((rows,), width, dtype=torch.int32, device=device)
        mask = valid_mask.to(device=device, dtype=torch.bool).reshape(rows, width)
        return mask.sum(-1, dtype=torch.int32).contiguous()

    result = lengths.to(device=device, dtype=torch.int32).reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"top-k lengths are empty; expected {rows} entries")
    if rows % result.numel() == 0 and result.numel() != rows:
        result = result.repeat_interleave(rows // result.numel())
    if result.numel() != rows:
        raise ValueError(
            f"top-k lengths have {result.numel()} entries; expected {rows}"
        )
    # FlashInfer has no useful interpretation for a count outside the static
    # index width.  Clamp rather than allowing an out-of-bounds scan; this is
    # especially important for graph warmup buffers whose values are filled in
    # place on a later replay.
    # The input can be a view into metadata shared by other consumers.
    result = result.clamp(min=0, max=width)
    if valid_mask is not None:
        mask = valid_mask.to(device=device, dtype=torch.bool).reshape(rows, width)
        prefix = torch.arange(width, device=device).unsqueeze(0) < result.unsqueeze(1)
        result = (mask & prefix).sum(-1, dtype=torch.int32)
    return result.contiguous()


def canonical_topk(
    indices: torch.Tensor,
    lengths: Optional[torch.Tensor],
    supported_widths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and canonicalize sparse indices for the fixed-width kernel.

    FlashInfer consumes a prefix of each row, as described by its length
    tensor.  Therefore invalid ``-1`` entries are stably moved to the right
    before the length is calculated.  This preserves the order of selected
    slots while preventing an interspersed padding entry from shortening the
    effective prefix and silently dropping a later selected slot.
    """
    indices = indices.to(torch.int32).contiguous()
    if indices.dim() != 2:
        raise ValueError(
            f"sparse top-k indices must be 2D [rows, width], got {indices.shape}"
        )
    rows, width = indices.shape
    if not supported_widths or any(int(value) <= 0 for value in supported_widths):
        raise ValueError(
            f"supported top-k widths must be positive, got {supported_widths}"
        )
    if width not in supported_widths:
        padded_width = next(
            (value for value in supported_widths if value >= width), None
        )
        if padded_width is None:
            raise RuntimeError(
                f"SM120 sparse MLA Top-K width {width} exceeds the largest "
                f"FlashInfer instantiation ({supported_widths[-1]})"
            )
        padded = torch.full(
            (rows, padded_width), -1, dtype=torch.int32, device=indices.device
        )
        padded[:, :width] = indices
        indices = padded

    width = indices.shape[-1]
    valid = indices >= 0
    # An explicit length denotes a prefix in the original layout.  Ignore
    # entries beyond it before compacting; otherwise stale values in the tail
    # could become live merely because an earlier slot was -1.
    normalized_lengths = (
        token_lens(lengths, rows, width, indices.device)
        if lengths is not None
        else None
    )
    if normalized_lengths is not None:
        prefix = torch.arange(width, device=indices.device).unsqueeze(0) < (
            normalized_lengths.unsqueeze(1)
        )
        valid &= prefix

    # Stable sort by validity (valid first) keeps selected-slot order while
    # producing a fixed-shape tensor suitable for CUDA graph replay.  Avoid
    # ``masked_select`` here: its dynamically sized allocation is not graph
    # safe when the number of valid slots changes between replays.
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    canonical = indices.gather(1, order)
    canonical_valid = valid.gather(1, order)
    canonical.masked_fill_(~canonical_valid, -1)
    effective_lengths = canonical_valid.sum(-1, dtype=torch.int32)
    return canonical, effective_lengths.contiguous()


def run_chunked_reference(
    *,
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    sinks: torch.Tensor,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
) -> None:
    """Correctness fallback for widths without a FlashInfer instantiation.

    The cache stays paged FP8.  At most ``rows * chunk_size`` selected entries
    are dequantized at a time, and an online softmax merges chunks and the
    learned sink.  All shapes and loop counts are static, so this path remains
    CUDA-graph capturable while avoiding a full ``[rows, topk, 512]`` buffer.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
        dequantize_slots_to_bf16,
    )

    if query.ndim != 3:
        raise ValueError(f"query must be [rows, heads, dim], got {query.shape}")
    rows, heads, dim = (int(v) for v in query.shape)
    if dim != 512:
        raise ValueError(f"SM120 sparse MLA requires head_dim=512, got {dim}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    q_fp32 = query.float()
    running_max = sinks.to(device=query.device, dtype=torch.float32).reshape(1, heads)
    if query.is_cuda and not torch.cuda.is_current_stream_capturing():
        if not torch.isfinite(running_max).all():
            raise ValueError("SM120 sparse MLA sinks must be finite")
    running_max = running_max.expand(rows, heads).clone()
    running_sum = torch.ones((rows, heads), dtype=torch.float32, device=query.device)
    running_out = torch.zeros(
        (rows, heads, dim), dtype=torch.float32, device=query.device
    )

    def merge_pool(
        cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        nonlocal running_max, running_sum, running_out
        flat_indices = indices.reshape(rows, -1).to(torch.int64)
        width = int(flat_indices.shape[1])
        lengths_i32 = token_lens(lengths, rows, width, query.device)
        columns = torch.arange(width, device=query.device).view(1, width)
        valid_all = (columns < lengths_i32.view(rows, 1)) & (flat_indices >= 0)
        for start in range(0, width, chunk_size):
            end = min(start + chunk_size, width)
            chunk_indices = flat_indices[:, start:end]
            valid = valid_all[:, start:end]
            kv = dequantize_slots_to_bf16(cache, chunk_indices.reshape(-1)).view(
                rows, end - start, dim
            )
            kv_fp32 = kv.float()
            logits = torch.einsum("rhd,rcd->rhc", q_fp32, kv_fp32).mul_(scale)
            logits.masked_fill_(~valid.unsqueeze(1), -torch.inf)
            chunk_max = logits.amax(dim=-1)
            next_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - next_max)
            weights = torch.exp(logits - next_max.unsqueeze(-1))
            weights.masked_fill_(~valid.unsqueeze(1), 0.0)
            running_out = running_out * old_scale.unsqueeze(-1) + torch.einsum(
                "rhc,rcd->rhd", weights, kv_fp32
            )
            running_sum = running_sum * old_scale + weights.sum(dim=-1)
            running_max = next_max

    merge_pool(swa_cache, swa_indices, swa_lens)
    if extra_cache is not None:
        if extra_indices is None or extra_lens is None:
            raise ValueError("extra cache requires both indices and lengths")
        merge_pool(extra_cache, extra_indices, extra_lens)
    out.copy_((running_out / running_sum.unsqueeze(-1)).to(out.dtype))


def run(
    *,
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    sinks: torch.Tensor,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
    is_prefill: bool = False,
) -> None:
    validate_sm120_head_count(int(query.shape[-2]))
    sm120_sparse_mla = _load_sm120_sparse_mla()
    if sm120_sparse_mla is None:
        raise RuntimeError(
            "SM120 sparse MLA kernel was unavailable during model initialization"
        )
    if is_prefill:
        _require_prefill_api(sm120_sparse_mla)

    kernel_query = query.contiguous()
    kernel_sinks = sinks.float()
    kernel_out = out
    original_heads = int(query.shape[-2])
    if original_heads == 8:
        kernel_query = torch.cat((kernel_query, torch.zeros_like(kernel_query)), dim=-2)
        kernel_sinks = torch.cat((kernel_sinks, torch.zeros_like(kernel_sinks)), dim=-1)
        kernel_out = torch.empty_like(kernel_query)
    sm120_sparse_mla(
        query=kernel_query,
        swa_kv_cache=swa_cache.unsqueeze(-2),
        workspace_buffer=workspace(query.device),
        sparse_indices=swa_indices,
        compressed_kv_cache=(
            extra_cache.unsqueeze(-2) if extra_cache is not None else None
        ),
        out=kernel_out,
        bmm1_scale=scale,
        sinks=kernel_sinks,
        kv_layout="NHD",
        swa_topk_lens=swa_lens,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lens,
        **({"is_prefill": True} if is_prefill else {}),
    )
    if kernel_out is not out:
        out.copy_(kernel_out[..., :original_heads, :])


def forward_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_topk_idxs: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    *,
    softmax_scale: float,
    sched_meta: Any = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
        remap_padding_to_first_entry,
    )

    batch, q_len, heads, dim = q.shape
    rows = batch * q_len
    swa_indices = topk_idxs.squeeze(2) if topk_idxs.dim() == 4 else topk_idxs
    extra_indices = extra_topk_idxs
    if extra_indices is not None and extra_indices.dim() == 4:
        extra_indices = extra_indices.squeeze(2)
    swa_width = int(swa_indices.reshape(rows, -1).shape[-1])
    extra_width = (
        int(extra_indices.reshape(rows, -1).shape[-1])
        if extra_indices is not None
        else 0
    )
    generic_fallback = uses_generic_fallback(swa_width, extra_width)
    swa_indices, swa_topk_lens = canonical_topk(
        swa_indices.reshape(rows, -1),
        topk_length,
        (swa_width,) if generic_fallback else SM120_SWA_TOPK_WIDTHS,
    )
    if extra_indices is not None:
        extra_indices, extra_topk_lens = canonical_topk(
            extra_indices.reshape(rows, -1),
            extra_topk_length,
            (extra_width,) if generic_fallback else SM120_EXTRA_TOPK_WIDTHS,
        )
    else:
        extra_topk_lens = None
    # FlashInfer still loads the V tile for positions its length tensor
    # masks (softmax weight 0). Point the canonical -1 tail at the row's
    # first live slot so a never-written pool slot cannot yield 0 * NaN,
    # matching the prefill remap. Lengths keep masking by position.
    swa_indices = remap_padding_to_first_entry(swa_indices)
    if extra_indices is not None:
        extra_indices = remap_padding_to_first_entry(extra_indices)
    # CUDA graph replay cannot allocate a 128 MiB workspace.  It must be
    # touched during eager warmup; workspace() raises if a new allocation
    # is attempted while capture is active.
    if not generic_fallback:
        warmup(q.device)
    # FlashInfer's SM120 ABI derives page size and block stride from the
    # cache tensor and requires page-64 (extra pool also page-2) layout.
    # The decode top-k space indexes the persistent pool's own geometry,
    # so the selected slots are gathered into a compact page-aligned
    # cache and the indices remapped before the kernel call.
    swa_decode_cache = kv_cache
    extra_decode_cache = (
        extra_k_cache
        if extra_k_cache is not None and extra_indices is not None
        else None
    )
    if not generic_fallback:
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            pack_slots_to_paged,
        )

        swa_decode_cache, swa_indices = pack_slots_to_paged(
            kv_cache, swa_indices, page_size=64, row_lens=swa_topk_lens
        )
        if extra_decode_cache is not None:
            page_size = 2 if int(extra_decode_cache.shape[1]) <= 2 else 64
            extra_decode_cache, extra_indices = pack_slots_to_paged(
                extra_decode_cache,
                extra_indices,
                page_size=page_size,
                row_lens=extra_topk_lens,
            )
    flat_q = q.reshape(batch * q_len, heads, dim).contiguous()
    flat_out = torch.empty_like(flat_q)
    sparse_runner = run_chunked_reference if generic_fallback else run
    sparse_runner(
        query=flat_q,
        swa_cache=swa_decode_cache,
        swa_indices=swa_indices,
        swa_lens=swa_topk_lens,
        extra_cache=extra_decode_cache,
        extra_indices=extra_indices,
        extra_lens=extra_topk_lens,
        out=flat_out,
        scale=softmax_scale,
        sinks=attn_sink.float().contiguous(),
    )
    return flat_out.view_as(q)
