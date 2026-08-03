"""CUDA-graph-safe non-finite diagnostics for DeepSeek-V4.

Set ``DSV4_NAN_DIAG=1`` before starting the server to add read-only detector
kernels to important DSV4 numerical boundaries. A detector prints at most one
structured event per ``(batch, source, layer)`` even when an entire tensor is
non-finite. Unmapped startup and CUDA-graph-capture batches use batch id zero
and are intentionally silent. Triton includes the winning ``pid (row, tile,
0)`` in the line.
The first printed integer is the model batch id, which joins to the host-side
``[DSV4_NAN_TRACE]`` line containing trace ids. The second integer is this
event payload for the first reported 256-element tile:

    source * 10^12 + layer(3) | first-column-in-tile(3) | n_nan(3) | n_inf(3)

For example, ``1017007001000`` means source 1, layer 17, first bad value at
tile offset 7, one NaN, and zero Inf values.

Source ids:
    1 = MoE activation input
    2 = router linear scores
    3 = router bias
    4 = context-parallel attention LSE
    5 = final hidden state
    6 = attention query
    7 = KV value before cache quantization/write
    8 = packed SWA KV cache read
    9 = packed compressed KV cache read
   10 = attention output

For an end-to-end service test only, a guarded injector is available. Both
variables are required, otherwise startup fails instead of silently changing
model data::

    DSV4_NAN_DIAG_TEST_INJECT=2,0,0
    DSV4_NAN_DIAG_TEST_INJECT_CONFIRM=I_UNDERSTAND_THIS_CHANGES_OUTPUT

The injection tuple is ``layer,row,col`` and writes one NaN into that layer's
MoE activation before the read-only detector runs.

Packed FP8 KV-cache reads have a separate guarded corruption injector for
end-to-end tests. ``layer,pool,kind`` accepts pool ``swa`` or ``compressed``
and kind ``fp8``, ``rope``, or ``scale``::

    DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE=2,swa,fp8
    DSV4_NAN_DIAG_TEST_INJECT_CONFIRM=I_UNDERSTAND_THIS_CHANGES_OUTPUT

The detector is a separate kernel: it never writes the inspected tensor or any
model output.  During CUDA graph capture it becomes part of the graph, so the
same check runs and reports on every replay without a host sync or ``.item()``.
"""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


ENABLED = os.environ.get("DSV4_NAN_DIAG", "0") == "1"

_TEST_INJECT_SPEC = os.environ.get("DSV4_NAN_DIAG_TEST_INJECT", "").strip()
_TEST_KV_CORRUPT_SPEC = os.environ.get(
    "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE", ""
).strip()
_TEST_INJECT_CONFIRM = os.environ.get("DSV4_NAN_DIAG_TEST_INJECT_CONFIRM", "").strip()
_TEST_INJECT_CONFIRM_VALUE = "I_UNDERSTAND_THIS_CHANGES_OUTPUT"


def _parse_test_inject_spec(spec: str) -> tuple[int, int, int] | None:
    if not spec:
        return None
    if not ENABLED:
        raise RuntimeError("DSV4_NAN_DIAG_TEST_INJECT requires DSV4_NAN_DIAG=1")
    if _TEST_INJECT_CONFIRM != _TEST_INJECT_CONFIRM_VALUE:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_INJECT is test-only; set "
            "DSV4_NAN_DIAG_TEST_INJECT_CONFIRM="
            f"{_TEST_INJECT_CONFIRM_VALUE} to acknowledge output mutation"
        )
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT must be layer,row,col, got " f"{spec!r}"
        )
    layer, row, col = (int(part) for part in parts)
    if min(layer, row, col) < 0:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT coordinates must be non-negative, "
            f"got {spec!r}"
        )
    return layer, row, col


TEST_INJECT = _parse_test_inject_spec(_TEST_INJECT_SPEC)

SOURCE_MOE_INPUT = 1
SOURCE_ROUTER_SCORES = 2
SOURCE_ROUTER_BIAS = 3
SOURCE_CP_ATTENTION_LSE = 4
SOURCE_FINAL_HIDDEN = 5
SOURCE_ATTENTION_QUERY = 6
SOURCE_KV_WRITE_INPUT = 7
SOURCE_SWA_KV_CACHE_READ = 8
SOURCE_COMPRESSED_KV_CACHE_READ = 9
SOURCE_ATTENTION_OUTPUT = 10

_KV_CORRUPT_POOL_TO_SOURCE = {
    "swa": SOURCE_SWA_KV_CACHE_READ,
    "compressed": SOURCE_COMPRESSED_KV_CACHE_READ,
}
_KV_CORRUPT_KIND = {"fp8": 1, "rope": 2, "scale": 3}


def _parse_test_kv_corrupt_spec(spec: str) -> tuple[int, int, int] | None:
    if not spec:
        return None
    if not ENABLED:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE requires DSV4_NAN_DIAG=1"
        )
    if _TEST_INJECT_CONFIRM != _TEST_INJECT_CONFIRM_VALUE:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE is test-only; set "
            "DSV4_NAN_DIAG_TEST_INJECT_CONFIRM="
            f"{_TEST_INJECT_CONFIRM_VALUE} to acknowledge cache mutation"
        )
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE must be layer,pool,kind, got "
            f"{spec!r}"
        )
    layer_text, pool, kind = parts
    layer = int(layer_text)
    if (
        layer < 0
        or pool not in _KV_CORRUPT_POOL_TO_SOURCE
        or kind not in _KV_CORRUPT_KIND
    ):
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE expects a non-negative layer, "
            "pool swa|compressed, and kind fp8|rope|scale; got "
            f"{spec!r}"
        )
    return layer, _KV_CORRUPT_POOL_TO_SOURCE[pool], _KV_CORRUPT_KIND[kind]


TEST_KV_CORRUPT = _parse_test_kv_corrupt_spec(_TEST_KV_CORRUPT_SPEC)

_BLOCK_N = 256
_DEFAULT_PRINTF_FIFO_MB = 64
_MAX_SOURCE_ID = 15
_MAX_LAYER_ID = 999
_STATE_LAYERS = _MAX_LAYER_ID + 2  # layer -1 (unscoped) plus layers 0..999
_PREWARMED_DEVICES: set[str] = set()
_BATCH_ID_TENSORS: dict[str, torch.Tensor] = {}
_LAST_REPORTED_BATCH_BY_DEVICE: dict[str, torch.Tensor] = {}
_REPORT_COUNT_BY_DEVICE: dict[str, torch.Tensor] = {}


if triton is not None:

    @triton.jit(do_not_specialize=["row", "col", "stride_row", "stride_col"])
    def _inject_nan_kernel(
        tensor_ptr,
        row,
        col,
        stride_row,
        stride_col,
    ):
        tl.store(
            tensor_ptr + row * stride_row + col * stride_col,
            float("nan"),
        )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "cols",
            "stride_row",
            "stride_col",
            "source_id",
            "layer_id",
            "state_index",
            "include_neg_inf",
        ]
    )
    def _report_nonfinite_tiles_kernel(
        tensor_ptr,
        batch_id_ptr,
        last_reported_batch_ptr,
        report_count_ptr,
        rows,
        cols,
        stride_row,
        stride_col,
        source_id,
        layer_id,
        state_index,
        include_neg_inf,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        col_start = tile * BLOCK_N
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = (row < rows) & (col_offsets < cols)
        values = tl.load(
            tensor_ptr + row * stride_row + col_offsets * stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        is_nan = mask & (values != values)
        is_pos_inf = mask & (values == float("inf"))
        is_neg_inf = mask & (values == -float("inf"))
        is_inf = is_pos_inf | (is_neg_inf & (include_neg_inf != 0))
        is_bad = is_nan | is_inf
        n_nan = tl.sum(is_nan.to(tl.int32), axis=0)
        n_inf = tl.sum(is_inf.to(tl.int32), axis=0)
        first_col = tl.min(
            tl.where(is_bad, col_offsets, cols),
            axis=0,
        )

        # device_print runs once per active Triton lane unless explicitly
        # guarded. Keep the scan vectorized, but let only CUDA thread 0 emit.
        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if (n_nan + n_inf > 0) & (thread_idx == 0):
            batch_id = tl.load(batch_id_ptr).to(tl.int64)
            # Batch zero has no host-side trace mapping. CUDA graph capture
            # and other untracked startup work use it deliberately so a
            # persistent failure cannot flood stderr before the service is
            # ready or consume the event for the first real request.
            if batch_id != 0:
                previous_batch = tl.atomic_xchg(
                    last_reported_batch_ptr + state_index,
                    batch_id,
                )
                if previous_batch != batch_id:
                    tl.atomic_add(report_count_ptr + state_index, 1)
                    first_offset = first_col - col_start
                    event = (
                        source_id.to(tl.int64) * 1_000_000_000_000
                        + layer_id.to(tl.int64) * 1_000_000_000
                        + first_offset * 1_000_000
                        + tl.minimum(n_nan, 999).to(tl.int64) * 1_000
                        + tl.minimum(n_inf, 999).to(tl.int64)
                    )
                    tl.device_print(
                        "[DSV4_NAN] batch,event=source(2d)layer(3d)"
                        "first_offset(3d)n_nan(3d)n_inf(3d):",
                        batch_id,
                        event,
                    )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "width",
            "q_len",
            "num_cache_blocks",
            "cache_block_size",
            "block_stride",
            "source_id",
            "layer_id",
            "state_index",
        ]
    )
    def _report_packed_fp8_kv_cache_kernel(
        cache_ptr,
        indices_ptr,
        lengths_ptr,
        batch_id_ptr,
        last_reported_batch_ptr,
        report_count_ptr,
        rows,
        width,
        q_len,
        num_cache_blocks,
        cache_block_size,
        block_stride,
        source_id,
        layer_id,
        state_index,
        HAS_LENGTHS: tl.constexpr,
        LENGTHS_PER_ROW: tl.constexpr,
        CORRUPT_KIND: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Scan the exact packed FP8 cache slots consumed by FlashMLA.

        The logical ``[..., 584]`` view is not token-contiguous inside a
        physical block. Bytes are laid out as all 576-byte token payloads,
        followed by all 8-byte scale payloads. This mirrors the production
        writer and FlashMLA reader rather than relying on tensor stride(1).
        """
        linear = tl.program_id(0).to(tl.int64)
        row = linear // width
        col = linear % width
        valid_width = width
        if HAS_LENGTHS:
            if LENGTHS_PER_ROW:
                length_idx = row
            else:
                length_idx = row // q_len
            valid_width = tl.load(lengths_ptr + length_idx).to(tl.int64)
        slot = tl.load(indices_ptr + linear).to(tl.int64)
        valid_slot = (
            (row < rows)
            & (col < valid_width)
            & (slot >= 0)
            & (slot < num_cache_blocks * cache_block_size)
        )

        block = slot // cache_block_size
        pos = slot % cache_block_size
        block_ptr = cache_ptr + block * block_stride
        token_data_ptr = block_ptr + pos * 576
        token_scale_ptr = block_ptr + cache_block_size * 576 + pos * 8

        # Test-only strong corruption. Every referenced instance of the
        # selected slot is assigned the same byte pattern, so duplicate top-k
        # indices are harmless and no auxiliary state/allocation is needed.
        if CORRUPT_KIND == 1:
            tl.store(token_data_ptr, 0x7F, mask=valid_slot)
        elif CORRUPT_KIND == 2:
            tl.store(token_data_ptr + 448, 0xC1, mask=valid_slot)
            tl.store(token_data_ptr + 449, 0x7F, mask=valid_slot)
        elif CORRUPT_KIND == 3:
            tl.store(token_scale_ptr, 0xFF, mask=valid_slot)

        lanes = tl.arange(0, BLOCK_N)

        # E4M3FN has exactly two NaN encodings: 0x7f and 0xff.
        fp8_byte = tl.load(
            token_data_ptr + lanes,
            mask=valid_slot & (lanes < 448),
            other=0,
        )
        bad_fp8 = valid_slot & (lanes < 448) & ((fp8_byte == 0x7F) | (fp8_byte == 0xFF))

        # The 64 RoPE elements are stored verbatim as little-endian BF16.
        rope_ptr = (token_data_ptr + 448).to(tl.pointer_type(tl.uint16))
        rope_bits = tl.load(
            rope_ptr + lanes,
            mask=valid_slot & (lanes < 64),
            other=0,
        )
        bad_rope = (
            valid_slot
            & (lanes < 64)
            & ((rope_bits & 0x7F80) == 0x7F80)
            & ((rope_bits & 0x007F) != 0)
        )

        # The eighth scale byte is padding. 0xff is NaN in E8M0FNU.
        scale_byte = tl.load(
            token_scale_ptr + lanes,
            mask=valid_slot & (lanes < 7),
            other=0,
        )
        bad_scale = valid_slot & (lanes < 7) & (scale_byte == 0xFF)

        n_fp8 = tl.sum(bad_fp8.to(tl.int32), axis=0)
        n_rope = tl.sum(bad_rope.to(tl.int32), axis=0)
        n_scale = tl.sum(bad_scale.to(tl.int32), axis=0)
        n_bad = n_fp8 + n_rope + n_scale
        first_bad_byte = tl.min(
            tl.where(
                bad_fp8,
                lanes,
                tl.where(
                    bad_rope,
                    448 + lanes * 2,
                    tl.where(bad_scale, 576 + lanes, 584),
                ),
            ),
            axis=0,
        )
        kind_bitmap = (
            (n_fp8 > 0).to(tl.int64)
            + (n_rope > 0).to(tl.int64) * 2
            + (n_scale > 0).to(tl.int64) * 4
        )

        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if (n_bad > 0) & (thread_idx == 0):
            batch_id = tl.load(batch_id_ptr).to(tl.int64)
            if batch_id != 0:
                previous_batch = tl.atomic_xchg(
                    last_reported_batch_ptr + state_index,
                    batch_id,
                )
                if previous_batch != batch_id:
                    tl.atomic_add(report_count_ptr + state_index, 1)
                    event = (
                        source_id.to(tl.int64) * 1_000_000_000_000
                        + layer_id.to(tl.int64) * 1_000_000_000
                        + tl.minimum(first_bad_byte, 999).to(tl.int64) * 1_000_000
                        + tl.minimum(n_bad, 999).to(tl.int64) * 1_000
                        + kind_bitmap
                    )
                    tl.device_print(
                        "[DSV4_NAN] batch,event=source(2d)layer(3d)"
                        "first_byte(3d)n_bad(3d)kind_bitmap(3d):",
                        batch_id,
                        event,
                    )
                    tl.device_print(
                        "[DSV4_KV_NAN] batch,slot:",
                        batch_id,
                        slot,
                    )
                    detail = (
                        source_id.to(tl.int64) * 1_000_000_000
                        + layer_id.to(tl.int64) * 1_000_000
                        + tl.minimum(n_fp8, 999).to(tl.int64) * 1_000
                        + tl.minimum(n_rope + n_scale, 999).to(tl.int64)
                    )
                    tl.device_print(
                        "[DSV4_KV_NAN] source_layer,n_fp8_n_rope_scale:",
                        detail,
                        n_rope.to(tl.int64) * 1_000 + n_scale.to(tl.int64),
                    )

    @triton.jit
    def _device_printf_canary_kernel():
        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if thread_idx == 0:
            tl.device_print("[DSV4_NAN_DIAG_CANARY] device-printf-ready:", 1)


def _normalized_device(device: str | torch.device) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def _report_state_index(source_id: int, layer_id: int) -> int:
    if not 0 <= source_id <= _MAX_SOURCE_ID:
        raise ValueError(
            f"NaN diagnostic source_id must be in [0, {_MAX_SOURCE_ID}], "
            f"got {source_id}"
        )
    if not -1 <= layer_id <= _MAX_LAYER_ID:
        raise ValueError(
            f"NaN diagnostic layer_id must be in [-1, {_MAX_LAYER_ID}], "
            f"got {layer_id}"
        )
    return source_id * _STATE_LAYERS + layer_id + 1


def _ensure_report_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    device = _normalized_device(device)
    device_key = str(device)
    last_reported = _LAST_REPORTED_BATCH_BY_DEVICE.get(device_key)
    report_count = _REPORT_COUNT_BY_DEVICE.get(device_key)
    if last_reported is not None and report_count is not None:
        return last_reported, report_count
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 NaN diagnostic state must be allocated before CUDA graph capture; "
            "call prewarm(device) first"
        )
    state_size = (_MAX_SOURCE_ID + 1) * _STATE_LAYERS
    last_reported = torch.full((state_size,), -1, dtype=torch.int64, device=device)
    report_count = torch.zeros((state_size,), dtype=torch.int64, device=device)
    _LAST_REPORTED_BATCH_BY_DEVICE[device_key] = last_reported
    _REPORT_COUNT_BY_DEVICE[device_key] = report_count
    return last_reported, report_count


def set_batch_context(batch_id: torch.Tensor | None) -> None:
    """Set the graph-stable batch id tensor used by detector launches."""
    if not ENABLED or batch_id is None:
        return
    if not batch_id.is_cuda or batch_id.dtype != torch.int64 or batch_id.numel() < 1:
        raise ValueError(
            "DSV4 NaN diagnostic batch id must be a non-empty CUDA int64 tensor"
        )
    device_key = str(_normalized_device(batch_id.device))
    _BATCH_ID_TENSORS[device_key] = batch_id


def report_nonfinite(
    tensor: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
    include_neg_inf: bool = True,
) -> None:
    """Report non-finite values without modifying or synchronizing ``tensor``."""
    if not ENABLED or tensor.numel() == 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    if not tensor.is_cuda:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires CUDA tensors")
    if not tensor.is_floating_point():
        raise ValueError(f"NaN diagnostic requires floating tensor, got {tensor.dtype}")
    state_index = _report_state_index(int(source_id), int(layer_id))
    if tensor.dim() == 1:
        rows, cols = 1, tensor.shape[0]
        stride_row, stride_col = 0, tensor.stride(0)
    elif tensor.dim() == 2:
        rows, cols = tensor.shape
        stride_row, stride_col = tensor.stride()
    elif tensor.dim() > 2 and tensor.is_contiguous():
        cols = tensor.shape[-1]
        rows = tensor.numel() // cols
        stride_row, stride_col = cols, 1
    else:
        raise ValueError(
            "NaN diagnostic supports 1D/2D tensors and contiguous higher-rank "
            f"tensors, got shape={tuple(tensor.shape)} contiguous={tensor.is_contiguous()}"
        )

    device = _normalized_device(tensor.device)
    device_key = str(device)
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        # Standalone diagnostic callers have no service trace map. Batch 0 is
        # the explicit "unmapped" value and is allocated before launch/capture.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN diagnostic batch context must be allocated before CUDA "
                "graph capture; call prewarm(device) first"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    last_reported_batch, report_count = _ensure_report_state(device)
    grid = (rows, triton.cdiv(cols, _BLOCK_N))
    _report_nonfinite_tiles_kernel[grid](
        tensor,
        batch_id_tensor,
        last_reported_batch,
        report_count,
        rows,
        cols,
        stride_row,
        stride_col,
        int(source_id),
        int(layer_id),
        state_index,
        int(include_neg_inf),
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )


def report_packed_fp8_kv_cache(
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
    topk_length: torch.Tensor | None = None,
) -> None:
    """Report NaN encodings in the packed FP8 slots read by attention.

    ``indices`` contains global slots into ``cache`` and must have shape
    ``[B, q_len, K]``. Only the leftmost ``topk_length[B]`` entries are
    scanned when a length tensor is supplied. The cache and indices are read
    in-place without materializing or dequantizing the selected KV values.
    """
    if not ENABLED or cache.numel() == 0 or indices.numel() == 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    if (
        not cache.is_cuda
        or cache.dtype != torch.uint8
        or cache.dim() != 3
        or cache.shape[-1] != 584
        or cache.stride(-1) != 1
    ):
        raise ValueError(
            "packed DSV4 KV diagnostic requires CUDA uint8 "
            f"[num_blocks, block_size, 584], got shape={tuple(cache.shape)} "
            f"dtype={cache.dtype} device={cache.device} stride={cache.stride()}"
        )
    if (
        not indices.is_cuda
        or indices.dtype != torch.int32
        or indices.dim() != 3
        or not indices.is_contiguous()
        or indices.device != cache.device
    ):
        raise ValueError(
            "packed DSV4 KV diagnostic requires contiguous CUDA int32 "
            f"indices [B, q_len, K] on {cache.device}, got "
            f"shape={tuple(indices.shape)} dtype={indices.dtype} "
            f"device={indices.device} contiguous={indices.is_contiguous()}"
        )
    width = int(indices.shape[-1])
    if width == 0:
        return
    rows = int(indices.numel() // width)
    q_len = int(indices.shape[-2])
    lengths_per_row = False
    if topk_length is not None:
        if (
            not topk_length.is_cuda
            or topk_length.dtype != torch.int32
            or not topk_length.is_contiguous()
            or topk_length.device != cache.device
        ):
            raise ValueError(
                "packed DSV4 KV diagnostic topk_length must be contiguous CUDA "
                f"int32 on {cache.device}, got dtype={topk_length.dtype} "
                f"device={topk_length.device} contiguous={topk_length.is_contiguous()}"
            )
        if topk_length.numel() == rows:
            lengths_per_row = True
        elif topk_length.numel() != rows // q_len:
            raise ValueError(
                "packed DSV4 KV diagnostic topk_length must have B or B*q_len "
                f"elements, got {topk_length.numel()} for indices "
                f"shape={tuple(indices.shape)}"
            )

    device = _normalized_device(cache.device)
    device_key = str(device)
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN diagnostic batch context must be allocated before CUDA "
                "graph capture; call prewarm(device) first"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    state_index = _report_state_index(int(source_id), int(layer_id))
    last_reported_batch, report_count = _ensure_report_state(device)
    corrupt_kind = 0
    if TEST_KV_CORRUPT is not None:
        corrupt_layer, corrupt_source, requested_kind = TEST_KV_CORRUPT
        if corrupt_layer == int(layer_id) and corrupt_source == int(source_id):
            corrupt_kind = requested_kind
    lengths_ptr = topk_length if topk_length is not None else indices
    _report_packed_fp8_kv_cache_kernel[(rows * width,)](
        cache,
        indices,
        lengths_ptr,
        batch_id_tensor,
        last_reported_batch,
        report_count,
        rows,
        width,
        q_len,
        int(cache.shape[0]),
        int(cache.shape[1]),
        int(cache.stride(0)),
        int(source_id),
        int(layer_id),
        state_index,
        HAS_LENGTHS=topk_length is not None,
        LENGTHS_PER_ROW=lengths_per_row,
        CORRUPT_KIND=corrupt_kind,
        BLOCK_N=512,
        num_warps=4,
        num_stages=1,
    )


def maybe_inject_test_nan(tensor: torch.Tensor, *, layer_id: int) -> None:
    """Inject one guarded test NaN; no-op unless TEST_INJECT targets this layer."""
    if TEST_INJECT is None or TEST_INJECT[0] != layer_id:
        return
    if triton is None:
        raise RuntimeError("DSV4 NaN test injection requires Triton")
    if not tensor.is_cuda or tensor.dim() != 2:
        raise RuntimeError(
            "DSV4 NaN test injection requires a 2D CUDA activation tensor"
        )
    _, row, col = TEST_INJECT
    if row >= tensor.shape[0] or col >= tensor.shape[1]:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT is outside the activation shape: "
            f"target=(layer={layer_id},row={row},col={col}) "
            f"shape={tuple(tensor.shape)}"
        )
    _inject_nan_kernel[(1,)](
        tensor,
        row,
        col,
        tensor.stride(0),
        tensor.stride(1),
        num_warps=1,
        num_stages=1,
    )


def prewarm(device: str | torch.device) -> None:
    """Compile BF16/FP32 detector variants before CUDA graph capture."""
    if not ENABLED:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    device = _normalized_device(device)
    device_key = str(device)
    if device_key in _PREWARMED_DEVICES:
        return

    fifo_mb = int(
        os.environ.get(
            "DSV4_NAN_DIAG_PRINTF_FIFO_MB",
            str(_DEFAULT_PRINTF_FIFO_MB),
        )
    )
    if fifo_mb <= 0:
        raise ValueError(
            f"DSV4_NAN_DIAG_PRINTF_FIFO_MB must be positive, got {fifo_mb}"
        )
    fifo_configured = True
    try:
        # CUDA's default device-printf FIFO is small enough to lose events
        # during a widespread NaN burst. This must run before any printf kernel.
        with torch.cuda.device(device):
            triton.runtime.driver.active.utils.set_printf_fifo_size(
                fifo_mb * 1024 * 1024
            )
    except Exception as error:
        fifo_configured = False
        logging.warning(
            "[DSV4 NaN diag] failed to set CUDA printf FIFO to %d MiB; "
            "continuing with the runtime FIFO size: %s",
            fifo_mb,
            error,
        )

    logging.warning(
        "[DSV4 NaN diag] enabled on %s with printf_fifo=%s; "
        "events are rate-limited per batch/source/layer; device logs use "
        "prefixes [DSV4_NAN] and [DSV4_NAN_DIAG_CANARY]. "
        "source_id: 1=moe_input, 2=router_scores, 3=router_bias, "
        "4=cp_attention_lse, 5=final_hidden, 6=attention_query, "
        "7=kv_write_input, 8=swa_kv_cache_read, "
        "9=compressed_kv_cache_read, 10=attention_output",
        device,
        f"{fifo_mb} MiB" if fifo_configured else "runtime-default",
    )
    if TEST_INJECT is not None:
        logging.error(
            "[DSV4 NaN diag] TEST-ONLY NaN injection is active: "
            "layer=%d row=%d col=%d; model output is intentionally mutated",
            *TEST_INJECT,
        )
    if TEST_KV_CORRUPT is not None:
        logging.error(
            "[DSV4 NaN diag] TEST-ONLY KV-cache corruption is active: "
            "layer=%d source=%d kind=%d; cache and output are intentionally mutated",
            *TEST_KV_CORRUPT,
        )
    _BATCH_ID_TENSORS[device_key] = torch.zeros((1,), dtype=torch.int64, device=device)
    _ensure_report_state(device)
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.zeros((1, _BLOCK_N), dtype=dtype, device=device)
        report_nonfinite(
            probe,
            source_id=SOURCE_ROUTER_SCORES,
            layer_id=-1,
        )
    packed_probe = torch.zeros((1, 1, 584), dtype=torch.uint8, device=device)
    index_probe = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    length_probe = torch.ones((1,), dtype=torch.int32, device=device)
    report_packed_fp8_kv_cache(
        packed_probe,
        index_probe,
        source_id=SOURCE_SWA_KV_CACHE_READ,
        layer_id=-1,
    )
    report_packed_fp8_kv_cache(
        packed_probe,
        index_probe,
        source_id=SOURCE_COMPRESSED_KV_CACHE_READ,
        layer_id=-1,
        topk_length=length_probe,
    )
    if TEST_INJECT is not None:
        probe = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
        _inject_nan_kernel[(1,)](
            probe,
            0,
            0,
            probe.stride(0),
            probe.stride(1),
            num_warps=1,
            num_stages=1,
        )
    _device_printf_canary_kernel[(1,)](num_warps=1, num_stages=1)
    torch.cuda.synchronize(device)
    _PREWARMED_DEVICES.add(device_key)
