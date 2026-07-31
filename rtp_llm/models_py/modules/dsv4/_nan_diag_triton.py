"""CUDA-graph-safe non-finite diagnostics for DeepSeek-V4.

Set ``DSV4_NAN_DIAG=1`` before starting the server to add read-only detector
kernels to DSV4 MoE inputs and router inputs. A detector prints one structured
event for each 256-element tile containing NaN or Inf. Triton includes the
``pid (row, tile, 0)`` in the line. The first printed integer is the model
batch id, which joins to the host-side ``[DSV4_NAN_TRACE]`` line containing
trace ids. The second integer is this 13-digit event payload:

    source(1 digit) | layer(3) | first-column-in-tile(3) | n_nan(3) | n_inf(3)

For example, ``1017007001000`` means source 1, layer 17, first bad value at
tile offset 7, one NaN, and zero Inf values.

Source ids:
    1 = MoE activation input
    2 = router linear scores
    3 = router bias

For an end-to-end service test only, a guarded injector is available. Both
variables are required, otherwise startup fails instead of silently changing
model data::

    DSV4_NAN_DIAG_TEST_INJECT=2,0,0
    DSV4_NAN_DIAG_TEST_INJECT_CONFIRM=I_UNDERSTAND_THIS_CHANGES_OUTPUT

The injection tuple is ``layer,row,col`` and writes one NaN into that layer's
MoE activation before the read-only detector runs.

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

_BLOCK_N = 256
_DEFAULT_PRINTF_FIFO_MB = 64
_PREWARMED_DEVICES: set[str] = set()
_BATCH_ID_TENSOR: torch.Tensor | None = None


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
        ]
    )
    def _report_nonfinite_tiles_kernel(
        tensor_ptr,
        batch_id_ptr,
        rows,
        cols,
        stride_row,
        stride_col,
        source_id,
        layer_id,
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
        is_inf = mask & (tl.abs(values) == float("inf"))
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
            first_offset = first_col - col_start
            event = (
                source_id.to(tl.int64) * 1_000_000_000_000
                + layer_id.to(tl.int64) * 1_000_000_000
                + first_offset * 1_000_000
                + tl.minimum(n_nan, 999).to(tl.int64) * 1_000
                + tl.minimum(n_inf, 999).to(tl.int64)
            )
            tl.device_print(
                "[DSV4_NAN] batch,event=source(1d)layer(3d)"
                "first_offset(3d)n_nan(3d)n_inf(3d):",
                batch_id,
                event,
            )


def set_batch_context(batch_id: torch.Tensor | None) -> None:
    """Set the graph-stable batch id tensor used by detector launches."""
    global _BATCH_ID_TENSOR
    if not ENABLED or batch_id is None:
        return
    if not batch_id.is_cuda or batch_id.dtype != torch.int64 or batch_id.numel() < 1:
        raise ValueError(
            "DSV4 NaN diagnostic batch id must be a non-empty CUDA int64 tensor"
        )
    _BATCH_ID_TENSOR = batch_id


def report_nonfinite(
    tensor: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
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
    if tensor.dim() == 1:
        rows, cols = 1, tensor.shape[0]
        stride_row, stride_col = 0, tensor.stride(0)
    elif tensor.dim() == 2:
        rows, cols = tensor.shape
        stride_row, stride_col = tensor.stride()
    else:
        raise ValueError(
            f"NaN diagnostic supports 1D/2D tensors, got shape={tuple(tensor.shape)}"
        )

    global _BATCH_ID_TENSOR
    if _BATCH_ID_TENSOR is None:
        # Standalone diagnostic callers have no service trace map. Batch 0 is
        # the explicit "unmapped" value and is allocated before launch/capture.
        _BATCH_ID_TENSOR = torch.zeros((1,), dtype=torch.int64, device=tensor.device)
    grid = (rows, triton.cdiv(cols, _BLOCK_N))
    _report_nonfinite_tiles_kernel[grid](
        tensor,
        _BATCH_ID_TENSOR,
        rows,
        cols,
        stride_row,
        stride_col,
        int(source_id),
        int(layer_id),
        BLOCK_N=_BLOCK_N,
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
    device = torch.device(device)
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
        "device logs use prefix [DSV4_NAN]. "
        "source_id: 1=moe_input, 2=router_scores, 3=router_bias",
        device,
        f"{fifo_mb} MiB" if fifo_configured else "runtime-default",
    )
    if TEST_INJECT is not None:
        logging.error(
            "[DSV4 NaN diag] TEST-ONLY NaN injection is active: "
            "layer=%d row=%d col=%d; model output is intentionally mutated",
            *TEST_INJECT,
        )
    global _BATCH_ID_TENSOR
    _BATCH_ID_TENSOR = torch.zeros((1,), dtype=torch.int64, device=device)
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.zeros((1, _BLOCK_N), dtype=dtype, device=device)
        report_nonfinite(
            probe,
            source_id=SOURCE_ROUTER_SCORES,
            layer_id=-1,
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
    torch.cuda.synchronize(device)
    _PREWARMED_DEVICES.add(device_key)
