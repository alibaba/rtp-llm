"""
COPIED FROM DeepGEMM
"""

import os
from typing import Tuple

import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_token_cast_back(
    x_fp8: torch.Tensor, x_scales: torch.Tensor, pertoken_quant: bool = False
):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align(n, 128)
    x_fp8_padded = torch.nn.functional.pad(
        x_fp8, (0, aligned_n - n), mode="constant", value=0
    )
    if x_scales.dtype == torch.int:
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2" or pertoken_quant:
            x_scales = x_scales << 23
        else:
            x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23

        x_scales = x_scales.view(dtype=torch.float)

    # L2 / pertoken: dequant from unpadded x_fp8 (element count m*n). When n % 128 != 0,
    # aligned_n > n; reshaping to x_fp8_padded.shape would require m*aligned_n elements
    # and raises RuntimeError — must fold back to x_fp8.shape (or reshape to x_fp8.shape).
    use_unpadded_dequant = (
        os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2" or pertoken_quant
    )
    if use_unpadded_dequant:
        x_fp32_groups = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, x_fp8.size(1))
    else:
        x_fp32_groups = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)

    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    dequant = x_fp32_groups * x_scales
    if use_unpadded_dequant:
        out_bf16 = dequant.reshape(x_fp8.shape).to(torch.bfloat16)
    else:
        out_bf16 = dequant.view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n]
    return out_bf16.contiguous()


def per_channel_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(0) % 128 == 0
    m, n = x.shape
    x_view = x.view(-1, 128, n)
    x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: Tuple, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int64).sum().item()


def assert_close_with_mismatch_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_mismatched_elements: int = 0,
):
    """
    Asserts that two tensors are close, allowing for a specified number of mismatched elements.
    NaN and Inf values count against the mismatch allowance.
    """
    # Ensure tensors are float for comparison
    actual_float = actual.float()
    expected_float = expected.float()

    non_finite = ~torch.isfinite(actual_float) | ~torch.isfinite(expected_float)
    mismatched = ~torch.isclose(
        actual_float,
        expected_float,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
    # torch.isclose considers equal infinities close. Keep all non-finite values
    # subject to max_mismatched_elements instead.
    mismatched |= non_finite

    num_mismatched = torch.sum(mismatched).item()

    if num_mismatched > max_mismatched_elements:
        # For a helpful error message, let's find the worst offenders
        actual_flat = actual_float.flatten()
        expected_flat = expected_float.flatten()
        finite = torch.isfinite(actual_flat) & torch.isfinite(expected_flat)
        if finite.any().item():
            finite_abs_diff = torch.abs(actual_flat[finite] - expected_flat[finite])
            finite_rel_diff = finite_abs_diff / (
                torch.abs(expected_flat[finite]) + 1e-12
            )
            greatest_abs_diff = f"{torch.max(finite_abs_diff).item():.4g}"
            greatest_rel_diff = f"{torch.max(finite_rel_diff).item():.4g}"
        else:
            greatest_abs_diff = "N/A"
            greatest_rel_diff = "N/A"

        total_elements = actual_flat.numel()
        num_non_finite = torch.sum(non_finite).item()

        raise AssertionError(
            f"Tensors are not close enough!\n"
            f"Mismatched elements: {num_mismatched} / {total_elements} "
            f"({100.0 * num_mismatched / total_elements:.2f}%)\n"
            f"Non-finite element pairs: {num_non_finite}\n"
            f"Allowed mismatched elements: {max_mismatched_elements}, but found {num_mismatched}.\n"
            f"Greatest finite absolute difference: {greatest_abs_diff} (atol={atol})\n"
            f"Greatest finite relative difference: {greatest_rel_diff} (rtol={rtol})"
        )
