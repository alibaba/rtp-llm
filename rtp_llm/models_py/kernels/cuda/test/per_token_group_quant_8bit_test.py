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
    # NUM_TOKENS = [127, 128, 512, 1024, 4096, 8192]
    # HIDDEN_DIMS = [256, 512, 1024, 2048, 4096]
    # GROUP_SIZES = [8, 16, 32, 64, 128]
    # DST_DTYPES = [torch.int8, fp8_type_]
    # COLUMN_MAJOR_SCALES = [False, True]
    # SCALE_TMA_ALIGNED = [False, True]
    NUM_TOKENS = [127]
    HIDDEN_DIMS = [256]
    GROUP_SIZES = [8]
    DST_DTYPES = [fp8_type_]
    COLUMN_MAJOR_SCALES = [False]
    SCALE_TMA_ALIGNED = [False]

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_quant_test(
        self,
        num_tokens,
        hidden_dim,
        group_size,
        dst_dtype,
        column_major_scales,
        scale_tma_aligned,
    ):
        if not column_major_scales and scale_tma_aligned:
            self.skipTest(
                "Invalid flag: scale_tma_aligned=True but column_major_scales=False"
            )
        torch.manual_seed(0)
        x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

        # 调用triton和sglang的量化算子
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
        print(f"x_q_triton = {x_q_triton}")
        print(f"x_s_triton = {x_s_triton}")
        print(f"x_q_sglang = {x_q_sglang}")
        print(f"x_s_sglang = {x_s_sglang}")

    def test_per_token_group_quant(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_DIMS,
            self.GROUP_SIZES,
            self.DST_DTYPES,
            self.COLUMN_MAJOR_SCALES,
            self.SCALE_TMA_ALIGNED,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_dim=params[1],
                group_size=params[2],
                dst_dtype=params[3],
                column_major_scales=params[4],
                scale_tma_aligned=params[5],
            ):
                self._run_quant_test(*params)


class DenseSiluGroupQuantTest(TestCase):
    """Bitwise regression against the production BF16 activation + v1 quantizer."""

    TARGET_TOKEN_COUNTS = (128, 256, 500, 512, 1000, 1024)
    TOKEN_COUNTS = TARGET_TOKEN_COUNTS + (131, 257, 503, 1009)
    HIDDEN_DIM = 3072

    def setUp(self):
        if _is_hip or not torch.cuda.is_available():
            raise SkipTest("Dense SiLU group quant is a CUDA path")
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v2, rtp_llm_ops

        self.make_scales = create_per_token_group_quant_fp8_output_scale
        self.quantize = sgl_per_token_group_quant_fp8
        self.quantize_v2 = per_token_group_quant_fp8_v2
        self.silu_and_mul = rtp_llm_ops.silu_and_mul

    def _outputs(self, x):
        shape = (x.shape[0], x.shape[1] // 2)
        q = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
        scales = self.make_scales(shape, x.device, 128, True, True, True)
        return q, scales

    def _separate(self, x, activated, q, scales):
        self.silu_and_mul(activated, x, torch.cuda.current_stream().cuda_stream)
        per_token_group_quant_fp8(activated, q, scales, 128, 1e-4, -448.0, 448.0, True)

    def _fused(self, x, q, scales):
        self.quantize_v2(x, q, scales, 128, 1e-4, -448.0, 448.0, True, True, None, True)

    def _assert_equal(self, x):
        expected_q, expected_scales = self._outputs(x)
        actual_q, actual_scales = self._outputs(x)
        activated = torch.empty_like(expected_q, dtype=torch.bfloat16)
        self._separate(x, activated, expected_q, expected_scales)
        self._fused(x, actual_q, actual_scales)
        self.assertTrue(
            torch.equal(actual_scales, expected_scales), "UE8M0 scale mismatch"
        )
        self.assertTrue(
            torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8)),
            "FP8 output bytes differ from the separate production path",
        )
        return expected_q, expected_scales

    def test_rejects_incomplete_hidden_groups(self):
        # These widths pass floor(hidden / 128) % 4 == 0 but contain a
        # partial group. M=128 also passes the total-numel divisibility check.
        # Call the raw binding to verify validation below the Python wrapper.
        for hidden in (513, 3073):
            with self.subTest(hidden=hidden):
                x = torch.empty((128, 2 * hidden), device="cuda", dtype=torch.bfloat16)
                q, scales = self._outputs(x)
                with self.assertRaisesRegex(RuntimeError, "hidden divisible by 512"):
                    self._fused(x, q, scales)

    def test_target_and_prime_shapes(self):
        for tokens in self.TOKEN_COUNTS:
            for seed in (0, 17, 20261015):
                with self.subTest(tokens=tokens, seed=seed):
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    x = torch.randn(
                        (tokens, 2 * self.HIDDEN_DIM),
                        generator=generator,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    reference_q, reference_scales = self._assert_equal(x)
                    actual_q, actual_scales = self.quantize(
                        x,
                        128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                        fuse_silu_and_mul=True,
                        preserve_silu_mul_rounding=True,
                    )
                    self.assertTrue(
                        torch.equal(
                            reference_q.view(torch.uint8), actual_q.view(torch.uint8)
                        )
                    )
                    self.assertTrue(torch.equal(reference_scales, actual_scales))

    def test_scale_and_fp8_rounding_boundaries(self):
        # SiLU(16) is close enough to 16 that the BF16 product reaches the
        # desired representable values. Sweep the adjacent BF16 values around
        # each UE8M0 amax boundary, then place FP8 rounding ties in each group.
        for tokens in self.TARGET_TOKEN_COUNTS:
            x = torch.empty(
                (tokens, 2 * self.HIDDEN_DIM), device="cuda", dtype=torch.bfloat16
            )
            x[:, : self.HIDDEN_DIM].fill_(16)
            up = x[:, self.HIDDEN_DIM :]
            up.zero_()
            for group in range(self.HIDDEN_DIM // 128):
                exponent = (-24, -20, -14, -1, 0, 1, 5, 10)[group % 8]
                boundary = torch.tensor(
                    448.0 * 2.0**exponent, device="cuda", dtype=torch.bfloat16
                )
                direction = group // 8 - 1
                if direction:
                    toward = torch.full_like(
                        boundary, float("inf") if direction > 0 else 0.0
                    )
                    boundary = torch.nextafter(boundary, toward)
                base = group * 128
                up[:, base] = boundary / 16
                tie = torch.tensor(
                    1.0625 * 2.0**exponent, device="cuda", dtype=torch.bfloat16
                )
                vals = (
                    torch.stack(
                        (
                            tie,
                            torch.nextafter(tie, torch.zeros_like(tie)),
                            torch.nextafter(tie, torch.full_like(tie, float("inf"))),
                            -tie,
                            -torch.nextafter(tie, torch.zeros_like(tie)),
                            -torch.nextafter(tie, torch.full_like(tie, float("inf"))),
                        )
                    )
                    / 16
                )
                up[:, base + 1 : base + 7] = vals
            self._assert_equal(x)
            x.zero_()
            self._assert_equal(x)
            # Nonlinear SiLU inputs include negative saturation and values
            # close to zero; vary the multiplier across several exponents.
            gate_values = torch.tensor(
                [-20, -8, -1, -(2**-10), 0, 2**-10, 1, 8, 20],
                dtype=torch.bfloat16,
                device="cuda",
            )
            columns = torch.arange(self.HIDDEN_DIM, device="cuda")
            x[:, : self.HIDDEN_DIM] = gate_values[columns % gate_values.numel()]
            x[:, self.HIDDEN_DIM :] = torch.exp2((columns % 17 - 8).float()).to(
                torch.bfloat16
            )
            self._assert_equal(x)

    def test_graph_replay_uses_new_input(self):
        for tokens in self.TARGET_TOKEN_COUNTS:
            x = torch.randn(
                (tokens, 2 * self.HIDDEN_DIM), device="cuda", dtype=torch.bfloat16
            )
            q, scales = self._outputs(x)
            self._fused(x, q, scales)
            torch.cuda.synchronize()
            pointers = (x.data_ptr(), q.data_ptr(), scales.data_ptr())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._fused(x, q, scales)
            for seed in (41, 73, 101):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                replacement = torch.randn(
                    x.shape, generator=generator, device=x.device, dtype=x.dtype
                )
                expected_q, expected_scales = self._assert_equal(replacement)
                x.copy_(replacement)
                graph.replay()
                self.assertEqual(
                    pointers, (x.data_ptr(), q.data_ptr(), scales.data_ptr())
                )
                self.assertTrue(
                    torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
                )
                self.assertTrue(torch.equal(scales, expected_scales))


if __name__ == "__main__":
    main()
