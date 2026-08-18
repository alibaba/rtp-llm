import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # CPU-only import
    triton = tl = None


ENABLED = os.environ.get("DSV4_NAN_DIAG", "0") == "1"
SOURCE_ATTENTION_INPUT, SOURCE_MOE_INPUT = 1, 3
_BLOCK_N, _EVENT_CAPACITY, _EVENT_FIELDS = 256, 4, 8
_STATES: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_PREWARMED: set[str] = set()
_DTYPE_IDS = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
}


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "rows",
            "cols",
            "stride_row",
            "stride_col",
            "first_source",
            "layer_id",
            "dtype_id",
        ]
    )
    def _report_kernel(
        tensor0,
        tensor1,
        tensor2,
        tensor3,
        state,
        events,
        rows,
        cols,
        stride_row,
        stride_col,
        first_source,
        layer_id,
        dtype_id,
        COUNT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAPACITY: tl.constexpr,
    ):
        row, tile, slot = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        ptr = tl.where(slot == 0, tensor0, tensor1)
        if COUNT == 4:
            ptr = tl.where(slot == 2, tensor2, ptr)
            ptr = tl.where(slot == 3, tensor3, ptr)
        offsets = tile.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row < rows) & (offsets < cols)
        values = tl.load(
            ptr + row.to(tl.int64) * stride_row + offsets * stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        is_nan = mask & (values != values)
        is_inf = mask & ((values == float("inf")) | (values == -float("inf")))
        n_nan = tl.sum(is_nan.to(tl.int32), axis=0)
        n_inf = tl.sum(is_inf.to(tl.int32), axis=0)
        first = tl.min(tl.where(is_nan | is_inf, offsets, cols), axis=0)
        thread = tl.inline_asm_elementwise(
            "mov.u32 $0, %tid.x;", "=r", [], tl.int32, is_pure=True, pack=1
        )
        source = first_source + slot
        if (n_nan + n_inf > 0) & (thread == 0):
            if tl.atomic_cas(state + source, 0, 1) == 0:
                event = tl.atomic_add(state, 1)
                if event < CAPACITY:
                    record = events + event * 8
                    tl.store(record + 0, source.to(tl.int64))
                    tl.store(record + 1, layer_id.to(tl.int64))
                    tl.store(record + 2, (row * cols + first).to(tl.int64))
                    tl.store(record + 3, n_nan.to(tl.int64))
                    tl.store(record + 4, n_inf.to(tl.int64))
                    tl.store(record + 5, dtype_id.to(tl.int64))
                    tl.store(record + 6, rows)
                    tl.store(record + 7, cols)


def _state(value: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = str(device)
    if key not in _STATES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prewarm DSV4 NaN diagnostics before CUDA graph capture")
        _STATES[key] = (
            torch.zeros(1 + _EVENT_CAPACITY, dtype=torch.int32, device=device),
            torch.empty(
                (_EVENT_CAPACITY, _EVENT_FIELDS), dtype=torch.int64, device=device
            ),
        )
    return _STATES[key]


def reset(device: str | torch.device) -> None:
    if ENABLED:
        _state(device)[0].zero_()


def attach_event_buffers(outputs):
    if ENABLED:
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None or not hidden.is_cuda:
            raise RuntimeError("DSV4 NaN diagnostics require CUDA model outputs")
        state, outputs.nan_diag_events = _state(hidden.device)
        outputs.nan_diag_event_counters = state[:1]
    return outputs


def _layout(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    if tensor.dim() == 1:
        return 1, tensor.shape[0], 0, tensor.stride(0)
    if tensor.dim() == 2:
        return *tensor.shape, *tensor.stride()
    if tensor.dim() > 2 and tensor.is_contiguous():
        cols = tensor.shape[-1]
        return tensor.numel() // cols, cols, cols, 1
    raise ValueError(f"unsupported DSV4 diagnostic shape: {tuple(tensor.shape)}")


def report(tensors: tuple[torch.Tensor, ...], first_source: int, layer_id: int) -> None:
    if not ENABLED or any(tensor.numel() == 0 for tensor in tensors):
        return
    rows, cols, stride_row, stride_col = _layout(tensors[0])
    padded = tensors + (tensors[-1],) * (4 - len(tensors))
    state, events = _state(tensors[0].device)
    _report_kernel[(rows, triton.cdiv(cols, _BLOCK_N), len(tensors))](
        *padded,
        state,
        events,
        rows,
        cols,
        stride_row,
        stride_col,
        first_source,
        layer_id,
        _DTYPE_IDS.get(tensors[0].dtype, 0),
        COUNT=len(tensors),
        BLOCK_N=_BLOCK_N,
        CAPACITY=_EVENT_CAPACITY,
        num_warps=4,
        num_stages=1,
    )


def prewarm(device: str | torch.device) -> None:
    if not ENABLED:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    device = _state(device)[0].device
    if str(device) in _PREWARMED:
        return
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.zeros((1, _BLOCK_N), dtype=dtype, device=device)
        report((probe,), SOURCE_ATTENTION_INPUT, -1)
        report((probe, probe, probe, probe), SOURCE_ATTENTION_INPUT, -1)
    torch.cuda.synchronize(device)
    reset(device)
    _PREWARMED.add(str(device))
