"""Numerical coverage for the optional CP FP8 row transport."""

import pytest
import torch

from rtp_llm.models_py.modules.dsv4.cp import (
    _fp8_row_dequant_kernel,
    _fp8_row_quant_kernel,
)


@pytest.mark.parametrize("width", [127, 1024, 2051])
def test_row_transport_is_finite_at_underflow_boundary(width):
    # Include zero lanes in each tiny row: an overflowing reciprocal produces
    # NaNs there even if nonzero lanes saturate to finite FP8 values.
    pattern = torch.tensor([0.0, 1.0, -1.0, 0.5], device="cuda")
    row = pattern.repeat((width + 3) // 4)[:width]
    magnitudes = torch.tensor(
        [0.0, 1e-38, 1e-37, 1e-35, 1e-4, 1.0, 1000.0], device="cuda"
    )
    values = (magnitudes[:, None] * row).to(torch.bfloat16)
    quantized = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scales = torch.empty(values.shape[0], device="cuda", dtype=torch.float32)
    restored = torch.empty_like(values)
    _fp8_row_quant_kernel[(values.shape[0],)](
        values, quantized, scales, width, BLOCK=1024
    )
    _fp8_row_dequant_kernel[(values.shape[0],)](
        quantized, scales, restored, width, BLOCK=1024
    )
    assert torch.isfinite(quantized.float()).all()
    assert torch.isfinite(scales.reciprocal()).all()
    assert torch.isfinite(restored).all()
    assert (restored[values == 0] == 0).all()
    # FP64 reference avoids the FP32 underflow being tested. Allow an FP8
    # rounding error and values below the device's normal FP32 range.
    reference = values.double()
    error = (restored.double() - reference).abs()
    tolerance = reference.abs() * 0.08 + torch.finfo(torch.float32).tiny
    assert (error <= tolerance).all()
