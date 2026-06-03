import itertools
from typing import Tuple
from unittest import SkipTest, TestCase, main

import torch
import triton
import triton.language as tl
from torch import dtype as _dtype
from torch.profiler import ProfilerActivity, profile, record_function

from rtp_llm.models_py.utils.arch import is_hip
from rtp_llm.ops.compute_ops import (
    per_token_group_quant_fp8,
    per_token_group_quant_int8,
)

_is_hip = is_hip()

fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_type_,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dtype == torch.int8:
        finfo = torch.iinfo(dtype)
    else:
        finfo = torch.finfo(dtype)

    fp8_max = finfo.max

    if _is_hip:
        if dtype == torch.int8:
            fp8_max = 127.0
        else:
            fp8_max = 224.0

    fp8_min = -fp8_max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_type_,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )

    if dtype == torch.int8:
        iinfo = torch.iinfo(dtype)
        int8_max = iinfo.max
        int8_min = iinfo.min
        per_token_group_quant_int8(
            x, x_q, x_s, group_size, eps, int8_min, int8_max, False
        )
    else:
        f8_info = torch.finfo(dtype)
        fp8_max = f8_info.max
        fp8_min = f8_info.min
        per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max, False)

    return x_q, x_s


class PerTokenGroupQuantTest(TestCase):
    NUM_TOKENS = [7]
    HIDDEN_DIMS = [256]
    GROUP_SIZES = [128, 256]
    INPUT_DTYPES = [torch.float16, torch.bfloat16]
    DST_DTYPES = [torch.int8, fp8_type_]
    SCALE_LAYOUTS = [
        (False, False),
        (True, False),
        (True, True),
    ]

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_quant_test(
        self,
        num_tokens,
        hidden_dim,
        group_size,
        input_dtype,
        dst_dtype,
        column_major_scales,
        scale_tma_aligned,
    ):
        torch.manual_seed(0)
        x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=input_dtype)

        x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(
            x,
            group_size,
            eps=1e-10,
            dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
        )

        x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(
            x,
            group_size,
            eps=1e-10,
            dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
        )

        torch.testing.assert_close(x_s_sglang, x_s_triton, rtol=1e-5, atol=1e-6)
        if dst_dtype == torch.int8:
            # BF16-to-int8 rounding can differ by one between CUDA and Triton.
            torch.testing.assert_close(x_q_sglang, x_q_triton, rtol=0, atol=1)
        else:
            self.assertTrue(
                torch.equal(x_q_sglang, x_q_triton),
                "quantized values differ from the Triton reference",
            )
        self.assertEqual(x_s_sglang.shape, x_s_triton.shape)
        self.assertEqual(x_s_sglang.stride(), x_s_triton.stride())

    def test_per_token_group_quant(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_DIMS,
            self.GROUP_SIZES,
            self.INPUT_DTYPES,
            self.DST_DTYPES,
            self.SCALE_LAYOUTS,
        ):
            column_major_scales, scale_tma_aligned = params[5]
            with self.subTest(
                num_tokens=params[0],
                hidden_dim=params[1],
                group_size=params[2],
                input_dtype=params[3],
                dst_dtype=params[4],
                column_major_scales=column_major_scales,
                scale_tma_aligned=scale_tma_aligned,
            ):
                self._run_quant_test(
                    *params[:5], column_major_scales, scale_tma_aligned
                )

    def test_reject_ue8m0_with_row_major_scales(self):
        x = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
        output_q = torch.empty_like(x, dtype=fp8_type_)
        output_s = torch.empty(2, 1, dtype=torch.int32, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, "scale_ue8m0 requires column-major output scales"
        ):
            per_token_group_quant_fp8(
                x, output_q, output_s, 128, 1e-10, -448.0, 448.0, True
            )


if __name__ == "__main__":
    main()
