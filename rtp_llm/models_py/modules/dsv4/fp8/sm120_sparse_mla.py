from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional, Sequence

import torch

from rtp_llm.models_py.utils.cuda_graph_warmup import cuda_graph_warmup_forward_enabled

# FlashInfer's SM120 split-K decode kernel serves at most this many query
# rows; larger calls always run a prefill kernel, whose shape envelope
# (page-64 primary cache, 64-multiple extra width) callers must then satisfy.
DECODE_MAX_TOKENS = 64

_sm120_sparse_mla = None
_sm120_sparse_mla_load_attempted = False
_SM120_SPARSE_MLA_LOCK = threading.Lock()


def _load_sm120_sparse_mla():
    """Load the FlashInfer SM120 sparse MLA entry point on first SM120 use.

    Device selection can import the adapter without loading the optional
    CUDA wheel. Provider validation happens when SM120 execution is prepared.
    """
    global _sm120_sparse_mla, _sm120_sparse_mla_load_attempted
    if _sm120_sparse_mla_load_attempted:
        return _sm120_sparse_mla
    with _SM120_SPARSE_MLA_LOCK:
        if _sm120_sparse_mla_load_attempted:
            return _sm120_sparse_mla
        loaded = None
        try:
            from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

            loaded = trtllm_batch_decode_sparse_mla_dsv4
        except (ImportError, AttributeError, OSError, RuntimeError) as error:
            logging.info("[dsv4-fp8] SM120 sparse MLA unavailable (%s)", error)
        _sm120_sparse_mla = loaded
        _sm120_sparse_mla_load_attempted = True
        return _sm120_sparse_mla


def sparse_mla_available() -> bool:
    return _load_sm120_sparse_mla() is not None


def prepare_decode(
    device: torch.device,
    n_heads: int,
    *,
    owner: Optional["Sm120SparseMlaWorkspace"] = None,
) -> None:
    """Validate and warm the provider on the device selected for decode."""
    validate_sm120_head_count(n_heads)
    if _load_sm120_sparse_mla() is None:
        raise RuntimeError(
            "SM120 FP8 sparse decode requires FlashInfer "
            "trtllm_batch_decode_sparse_mla_dsv4; install the CUDA 13 "
            "RTP-LLM dependency set before starting the service"
        )
    workspace(device, owner=owner)


_DEFAULT_MAX_EAGER_STREAMS = 8


class Sm120SparseMlaWorkspace:
    """Owner-scoped FlashInfer SM120 sparse-MLA scratch.

    Each independent model or runner instance owns a dedicated graph scratch
    slot ``(device_type, device_index, None)`` whose replay is serialized by
    CudaGraphRunner, avoiding cross-owner workspace aliasing. Eager execution
    uses per-stream slots bounded per device by LRU eviction so dynamic
    streams cannot pin unbounded memory while never evicting graph scratch.
    """

    def __init__(
        self,
        *,
        max_eager_streams: int = _DEFAULT_MAX_EAGER_STREAMS,
        allow_graph: bool = True,
    ):
        if int(max_eager_streams) <= 0:
            raise ValueError(
                f"max_eager_streams must be positive, got {max_eager_streams}"
            )
        self._buffers: OrderedDict[
            tuple[str, Optional[int], Optional[int]], torch.Tensor
        ] = OrderedDict()
        self._lock = threading.Lock()
        self._max_eager_streams = int(max_eager_streams)
        self._allow_graph = bool(allow_graph)

    def get(
        self, device: torch.device, *, for_graph: Optional[bool] = None
    ) -> torch.Tensor:
        key, capturing = _workspace_key(device, for_graph=for_graph)
        if key[2] is None and not self._allow_graph:
            raise RuntimeError(
                "SM120 sparse MLA CUDA graph capture/warmup requires an explicit "
                "Sm120SparseMlaWorkspace owner; the process-global default workspace "
                "is eager-only to prevent cross-graph scratch aliasing"
            )
        with self._lock:
            result = self._buffers.get(key)
            if result is not None:
                self._buffers.move_to_end(key)
                return result
            if capturing:
                raise RuntimeError(
                    "SM120 sparse MLA workspace must be materialized before "
                    "CUDA graph capture"
                )
            result = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._buffers[key] = result
            if key[2] is not None:
                device_prefix = key[:2]
                eager_keys = [
                    existing
                    for existing in self._buffers
                    if existing[:2] == device_prefix and existing[2] is not None
                ]
                overflow = len(eager_keys) - self._max_eager_streams
                for stale_key in eager_keys[: max(0, overflow)]:
                    del self._buffers[stale_key]
            return result

    def warmup(self, device: torch.device) -> torch.Tensor:
        return self.get(device, for_graph=True)


_DEFAULT_WORKSPACE = Sm120SparseMlaWorkspace(allow_graph=False)
_WORKSPACES = _DEFAULT_WORKSPACE._buffers

# Head counts with a native FlashInfer SM120 sparse-MLA instantiation.
SM120_HEAD_COUNTS = (8, 16, 32, 64, 128)


def validate_sm120_head_count(heads: int) -> None:
    if heads not in SM120_HEAD_COUNTS:
        raise ValueError(
            "SM120 sparse MLA requires 8, 16, 32, 64 or 128 local query heads; "
            f"got {heads}. Reduce tensor parallelism or use a supported head count."
        )


# FlashInfer's DSV4 sparse-decode dispatcher specializes on the static index
# width.  HCA compresses one entry per 128 input tokens, so a 1M context needs
# an 8192-wide extra sparse table.  Keep the complete production width set in
# one place so eager and CUDA-graph preparation cannot diverge.
SM120_SWA_TOPK_WIDTHS = (128, 512, 1024)
# Native prefill reads full 64-entry tiles; narrow decode supports width 2.
SM120_PREFILL_EXTRA_TOPK_WIDTHS = (128, 512, 1024, 2048, 4096, 8192)
SM120_EXTRA_TOPK_WIDTHS = (2, *SM120_PREFILL_EXTRA_TOPK_WIDTHS)
_DUAL_PREFILL_SWA_TOPK_WIDTH = 128


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


def uses_generic_fallback(
    swa_width: int, extra_width: int = 0, *, is_dual_prefill: bool = False
) -> bool:
    """Whether a sparse shape needs the graph-safe chunked implementation."""
    max_swa_width = (
        _DUAL_PREFILL_SWA_TOPK_WIDTH if is_dual_prefill else SM120_SWA_TOPK_WIDTHS[-1]
    )
    return swa_width > max_swa_width or extra_width > SM120_EXTRA_TOPK_WIDTHS[-1]


def _is_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if device.index is None:
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


def _workspace_key(
    device: torch.device, *, for_graph: Optional[bool] = None
) -> tuple[tuple[str, Optional[int], Optional[int]], bool]:
    cuda_available = torch.cuda.is_available()
    capturing = _is_stream_capturing(device)
    graph_active = capturing or cuda_graph_warmup_forward_enabled()
    if for_graph is False and graph_active:
        raise RuntimeError(
            "Cannot request an eager SM120 sparse MLA workspace during CUDA "
            "graph capture or warmup"
        )
    use_graph = graph_active if for_graph is None else bool(for_graph)
    device_index = (
        (torch.cuda.current_device() if device.index is None else device.index)
        if device.type == "cuda"
        else device.index
    )
    stream_id = (
        None
        if use_graph
        else (
            torch.cuda.current_stream(device).cuda_stream
            if device.type == "cuda" and cuda_available
            else 0
        )
    )
    return (device.type, device_index, stream_id), capturing


def workspace(
    device: torch.device,
    *,
    for_graph: Optional[bool] = None,
    owner: Optional[Sm120SparseMlaWorkspace] = None,
) -> torch.Tensor:
    target = owner if owner is not None else _DEFAULT_WORKSPACE
    return target.get(device, for_graph=for_graph)


def warmup(
    device: torch.device,
    *,
    owner: Optional[Sm120SparseMlaWorkspace] = None,
) -> torch.Tensor:
    """Materialize the fixed FlashInfer workspace before graph capture."""
    return workspace(device, for_graph=True, owner=owner)


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
    *,
    max_slot: Optional[int] = None,
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
    if max_slot is not None:
        valid &= indices < int(max_slot)
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

    if not sinks.is_cuda and not torch.isfinite(sinks.float()).all():
        raise ValueError("SM120 sparse MLA sinks must be finite")
    q_fp32 = query.float()
    running_max = (
        sinks.to(device=query.device, dtype=torch.float32)
        .reshape(1, heads)
        .expand(rows, heads)
        .clone()
    )
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
        pool_capacity = int(cache.shape[0] * cache.shape[1]) if cache.dim() >= 2 else None
        valid_all = (columns < lengths_i32.view(rows, 1)) & (flat_indices >= 0)
        if pool_capacity is not None:
            valid_all = valid_all & (flat_indices < pool_capacity)
        for start in range(0, width, chunk_size):
            end = min(start + chunk_size, width)
            chunk_indices = flat_indices[:, start:end]
            valid = valid_all[:, start:end]
            safe_chunk_indices = chunk_indices.masked_fill(~valid, -1).reshape(-1)
            kv = dequantize_slots_to_bf16(cache, safe_chunk_indices).view(
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
    workspace_owner: Optional[Sm120SparseMlaWorkspace] = None,
) -> None:
    """Run FlashInfer's SM120 DSv4 sparse MLA into ``out``.

    FlashInfer selects the decode or prefill kernel from the row count; it has
    no phase argument. ``is_prefill`` marks a prefill caller, whose shapes are
    held to the prefill envelope at every chunk size so that admission does
    not depend on how many rows a chunk happens to contain.
    """
    validate_sm120_head_count(int(query.shape[-2]))
    sm120_sparse_mla = _load_sm120_sparse_mla()
    if sm120_sparse_mla is None:
        raise RuntimeError(
            "SM120 sparse MLA kernel was unavailable during model initialization"
        )
    if is_prefill or int(query.shape[0]) > DECODE_MAX_TOKENS:
        if swa_cache.ndim >= 2 and int(swa_cache.shape[1]) != 64:
            raise ValueError(
                "Unsupported sparse-MLA prefill configuration: "
                f"page_block_size={int(swa_cache.shape[1])} (expected 64)"
            )
        if extra_indices is not None and int(extra_indices.shape[-1]) % 64 != 0:
            raise ValueError(
                "prefill extra_indices width must be a multiple of 64, "
                f"got {int(extra_indices.shape[-1])}"
            )

    kernel_query = query.contiguous()
    kernel_sinks = sinks.float()
    kernel_out = out
    original_heads = int(query.shape[-2])
    with torch.cuda.device(query.device):
        if original_heads == 8:
            kernel_query = torch.cat((kernel_query, torch.zeros_like(kernel_query)), dim=-2)
            kernel_sinks = torch.cat((kernel_sinks, torch.zeros_like(kernel_sinks)), dim=-1)
            kernel_out = torch.empty_like(kernel_query)
        sm120_sparse_mla(
            query=kernel_query,
            swa_kv_cache=swa_cache.unsqueeze(-2),
            workspace_buffer=workspace(query.device, owner=workspace_owner),
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
    workspace_owner: Optional[Sm120SparseMlaWorkspace] = None,
) -> torch.Tensor:
    from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
        remap_padding_to_first_entry,
    )

    batch, q_len, heads, dim = q.shape
    rows = batch * q_len
    if rows == 0:
        return torch.empty_like(q)
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
    extra_decode_cache = extra_k_cache if extra_indices is not None else None
    # Packed native decode supplies page-64 SWA and supported head and Top-K
    # shapes, so only the row limit sends these dual-cache calls to prefill.
    is_dual_prefill = rows > DECODE_MAX_TOKENS and extra_decode_cache is not None
    generic_fallback = uses_generic_fallback(
        swa_width, extra_width, is_dual_prefill=is_dual_prefill
    )
    swa_pool_capacity = int(kv_cache.shape[0] * kv_cache.shape[1])
    swa_indices, swa_topk_lens = canonical_topk(
        swa_indices.reshape(rows, -1),
        topk_length,
        (swa_width,) if generic_fallback else SM120_SWA_TOPK_WIDTHS,
        max_slot=swa_pool_capacity,
    )
    if extra_indices is not None:
        extra_widths = SM120_EXTRA_TOPK_WIDTHS
        if generic_fallback and extra_width:
            extra_widths = (extra_width,)
        elif is_dual_prefill and not generic_fallback:
            extra_widths = SM120_PREFILL_EXTRA_TOPK_WIDTHS
        extra_pool_capacity = (
            int(extra_decode_cache.shape[0] * extra_decode_cache.shape[1])
            if extra_decode_cache is not None
            else None
        )
        extra_indices, extra_topk_lens = canonical_topk(
            extra_indices.reshape(rows, -1),
            extra_topk_length,
            extra_widths,
            max_slot=extra_pool_capacity,
        )
    else:
        extra_topk_lens = None
    # FlashInfer's SM120 ABI derives page size and block stride from the
    # cache tensor and requires page-64 (extra pool also page-2) layout.
    # The decode top-k space indexes the persistent pool's own geometry,
    # so the selected slots are gathered into a compact page-aligned
    # cache and the indices remapped before the kernel call.
    swa_decode_cache = kv_cache
    if not generic_fallback:
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            pack_slots_to_paged,
        )

        # FlashInfer still loads the V tile for positions its length tensor
        # masks (softmax weight 0). Point the canonical -1 tail at the row's
        # first live slot so a never-written pool slot cannot yield 0 * NaN,
        # matching the prefill remap. Lengths keep masking by position.
        swa_indices = remap_padding_to_first_entry(swa_indices)
        if extra_indices is not None:
            extra_indices = remap_padding_to_first_entry(extra_indices)

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
        **({} if generic_fallback else {"workspace_owner": workspace_owner}),
    )
    return flat_out.view_as(q)
