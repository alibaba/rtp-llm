# type: ignore
import enum
import random
import time
from typing import Generator, List, Tuple
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.utils.deepgemm_wrapper import (
    bf16_gemm_nt,
    fp8_gemm_nt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.test.utils.bench_util import bench_kineto, calc_diff, count_bytes


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


class KernelType(enum.Enum):
    # For SM100 GEMMs
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


def get_arch_major() -> int:
    major, _ = torch.cuda.get_device_capability()
    return major


def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()


def get_kernel_types(use_bf16: bool = False) -> Tuple[KernelType, ...]:
    if use_bf16:
        return (KernelType.KernelNoSF,)
    return (
        (KernelType.Kernel1D2D,)
        if get_arch_major() == 9
        else (KernelType.Kernel1D1D, KernelType.Kernel1D2D)
    )


def get_out_dtype() -> Tuple[torch.dtype, ...]:
    return (torch.bfloat16,) if get_arch_major() == 9 else (torch.bfloat16, torch.float)


def get_major_ab() -> Tuple[Tuple[MajorTypeAB, MajorTypeAB], ...]:
    return ((MajorTypeAB.KMajor, MajorTypeAB.KMajor),)


def enumerate_normal(
    use_bf16: bool = False,
) -> Generator[Tuple[KernelType, int, int, int, bool, torch.dtype], None, None]:
    for kernel_type in get_kernel_types(use_bf16):
        for m in (128, 4096):
            for n, k in [
                (2112, 7168),
                (24576, 1536),
                (32768, 512),
                (7168, 16384),
                (4096, 7168),
                (7168, 2048),
            ]:
                for out_dtype in get_out_dtype():
                    for accumulate in (
                        (False,)
                        if out_dtype == torch.bfloat16 or kernel_type.is_1d2d()
                        else (False, True)
                    ):
                        yield kernel_type, m, n, k, accumulate, out_dtype


def enumerate_m_grouped_contiguous(
    use_bf16: bool = False,
) -> Generator[Tuple[KernelType, int, int, int, int], None, None]:
    for kernel_type in get_kernel_types(use_bf16):
        for num_groups, expected_m_per_group, n, k in (
            (4, 8192, 4096, 7168),
            (4, 8192, 7168, 2048),
            (8, 4096, 4096, 7168),
            (8, 4096, 7168, 2048),
        ):
            yield kernel_type, num_groups, expected_m_per_group, n, k


def enumerate_m_grouped_masked() -> (
    Generator[Tuple[KernelType, int, int, int, int, int], None, None]
):
    max_m = 4096
    for kernel_type in get_kernel_types():
        for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
            for n, k in (
                (4096, 7168),
                (7168, 2048),
            ):
                yield kernel_type, num_groups, max_m, m, n, k


def enumerate_sf_layout() -> Generator[Tuple[int, int, bool, bool, int], None, None]:
    for use_ue8m0 in (False, True):
        for with_transpose in (True, False):
            for mn in (4096, 4097, 8192):
                for k in (128, 7168, 7296):
                    for num_groups in (1, 2, 4):
                        yield mn, k, with_transpose, use_ue8m0, num_groups


def generate_normal(
    m: int,
    n: int,
    k: int,
    accumulate: bool,
    out_dtype: torch.dtype,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    d = (
        torch.randn((m, n), device="cuda", dtype=out_dtype) * 32
        if accumulate
        else torch.empty((m, n), device="cuda", dtype=out_dtype)
    )
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        return a, b, c, d, ref_d

    a_fp8, b_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0), per_block_cast_to_fp8(
        b, use_ue8m0=use_ue8m0
    )
    return a_fp8, b_fp8, c, d, ref_d


def get_mk_alignment_for_contiguous_layout():
    return 128


def generate_m_grouped_contiguous(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actual_ms = [
        int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)
    ]
    aligned_ms = [
        align(actual_m, get_mk_alignment_for_contiguous_layout())
        for actual_m in actual_ms
    ]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        return m, a, b, m_indices, d, ref_d

    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    return m, a_fp8, b_fp8, m_indices, d, ref_d


def generate_m_grouped_masked(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.einsum("gmk,gnk->gmn", a, b)

    masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, max_m, ceil_div(k, 128)), device="cuda", dtype=torch.float
        ),
    )
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i], use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)

    return a_fp8, b_fp8, masked_m, d, ref_d


def generate_k_grouped_contiguous(
    num_groups: int, m: int, n: int, ks: List[int], use_ue8m0: bool
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert get_mk_alignment_for_contiguous_layout() % 128 == 0
    k = sum(ks)

    a = torch.randn((k, m), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device="cuda", dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    start = 0
    for i, group_k in enumerate(ks):
        end = start + group_k
        ref_d[i] = c[i] + (a[start:end].T @ b[start:end])
        start = end

    a_fp8 = per_channel_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_channel_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    return k, a_fp8, b_fp8, c, d, ref_d


class DeepGEMMTest(TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_fp8_gemm_nt(self) -> None:
        print("Testing FP8 GEMM NT:", flush=True)
        for kernel_type, m, n, k, accumulate, out_dtype in enumerate_normal():
            out_opt = "FP32" if out_dtype == torch.float else "BF16"
            acc_opt = f"acc={int(accumulate)}"
            kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0

            for test_alias in (False, True):
                a, b, c, d, ref_d = generate_normal(
                    m, n, k, accumulate, out_dtype, use_ue8m0=use_ue8m0
                )
                if test_alias:
                    assert a[0].is_contiguous() and b[0].is_contiguous()
                fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, (
                    f"{m=}, {n=}, {k=}, {kernel_opt}, {accumulate=}, {out_dtype=}, "
                    f"{diff:.5f}, alias={test_alias}"
                )
            a, b, c, d, ref_d = generate_normal(
                m, n, k, accumulate, out_dtype, use_ue8m0=use_ue8m0
            )

            # Test launch overhead
            launch_start_t = time.time_ns()
            fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
            launch_end_t = time.time_ns()
            torch.cuda.synchronize()

            # noinspection PyShadowingNames
            def test_func():
                fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)

            t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
            print(
                f" > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, {out_opt}, {acc_opt}): "
                f"launch {(launch_end_t - launch_start_t) / 1e3:4.0f} us | {t * 1e6:4.0f} us | "
                f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s",
                flush=True,
            )
        print(flush=True)

    def test_m_grouped_fp8_gemm_nt_contiguous(self) -> None:
        print("Testing m-grouped contiguous FP8 GEMM NT:", flush=True)

        for (
            kernel_type,
            num_groups,
            expected_m_per_group,
            n,
            k,
        ) in enumerate_m_grouped_contiguous():
            kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0

            for test_alias in (False, True):
                m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                    num_groups, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
                )
                if test_alias:
                    assert a[0].is_contiguous() and b[0].is_contiguous()
                m_grouped_fp8_gemm_nt_contiguous(
                    a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
                )
                d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
                diff = calc_diff(d, ref_d)
                assert (
                    diff < 0.001
                ), f"{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}, alias={test_alias}"
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
            )

            # noinspection PyShadowingNames
            def test_func():
                m_grouped_fp8_gemm_nt_contiguous(
                    a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
                )

            t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
            print(
                f" > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): "
                f"{t * 1e6:4.0f} us | "
                f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s",
                flush=True,
            )
        print(flush=True)

    def test_m_grouped_fp8_gemm_nt_masked(self) -> None:
        print("Testing m-grouped masked FP8 GEMM NT:", flush=True)

        # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
        for (
            kernel_type,
            num_groups,
            max_m,
            expected_m_per_group,
            n,
            k,
        ) in enumerate_m_grouped_masked():
            kernel_opt = f"1D1D" if kernel_type.is_1d1d() else "1D2D"
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0

            # Test correctness
            for i in range(10):
                a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                    num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
                )
                m_grouped_fp8_gemm_nt_masked(
                    a,
                    b,
                    d,
                    masked_m,
                    expected_m_per_group,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )
                for j in range(num_groups):
                    diff = calc_diff(
                        d[j, : masked_m[j].item()], ref_d[j, : masked_m[j].item()]
                    )
                    assert (
                        diff < 0.001
                    ), f"{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}"

            # Construct full cases
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
            )

            # noinspection PyShadowingNames
            def test_func():
                m_grouped_fp8_gemm_nt_masked(
                    a,
                    b,
                    d,
                    masked_m,
                    expected_m_per_group,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )

            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
            print(
                f" > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): "
                f"{t * 1e6:4.0f} us | "
                f"{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s",
                flush=True,
            )
        print(flush=True)

    def test_bf16_gemm_nt(self) -> None:
        print("Testing BF16 GEMM NT:", flush=True)
        for _, m, n, k, accumulate, out_dtype in enumerate_normal(use_bf16=True):
            out_opt = "FP32" if out_dtype == torch.float else "BF16"
            acc_opt = f"acc={int(accumulate)}"

            for test_alias in (False, True):
                a, b, c, d, ref_d = generate_normal(
                    m, n, k, accumulate, out_dtype, use_bf16=True
                )
                if test_alias:
                    assert a.is_contiguous() and b.is_contiguous()
                bf16_gemm_nt(a, b, d, c=c)
                diff = calc_diff(d, ref_d)
                assert diff < 0.0001, (
                    f"{m=}, {n=}, {k=}, {accumulate=}, {out_dtype=}, "
                    f"{diff:.5f}, alias={test_alias}"
                )
            a, b, c, d, ref_d = generate_normal(
                m, n, k, accumulate, out_dtype, use_bf16=True
            )

            cublas_t = 0
            t = bench_kineto(
                lambda: bf16_gemm_nt(a, b, d, c=c),
                "bf16_gemm",
                suppress_kineto_output=True,
            )
            if accumulate == 0 and out_dtype == torch.bfloat16:
                # noinspection PyBroadException
                try:
                    cublas_t = bench_kineto(
                        lambda: a @ b.T, "nvjet", suppress_kineto_output=True
                    )
                except Exception:
                    pass
            print(
                f" > Perf (m={m:5}, n={n:5}, k={k:5}, {out_opt}, {acc_opt}): "
                f"{t * 1e6:4.0f} us | "
                f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | "
                f"{cublas_t / t:.2f}x cuBLAS",
                flush=True,
            )
        print(flush=True)

    def test_m_grouped_bf16_gemm_nt_contiguous(self) -> None:
        print("Testing m-grouped contiguous BF16 GEMM NT:", flush=True)

        for _, num_groups, expected_m_per_group, n, k in enumerate_m_grouped_contiguous(
            use_bf16=True
        ):
            for test_alias in (False, True):
                m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                    num_groups, expected_m_per_group, n, k, use_bf16=True
                )
                if test_alias:
                    assert a.is_contiguous() and b.is_contiguous()
                m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)
                d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f"{m=}, {n=}, {k=}, {diff:.5f}, alias={test_alias}"
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups, expected_m_per_group, n, k, use_bf16=True
            )

            # noinspection PyShadowingNames
            def test_func():
                m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)

            t = bench_kineto(test_func, "bf16_gemm", suppress_kineto_output=True)
            print(
                f" > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}): "
                f"{t * 1e6:4.0f} us | "
                f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s",
                flush=True,
            )
        print(flush=True)

    def test_m_grouped_bf16_gemm_nt_masked(self) -> None:
        print("Testing m-grouped masked BF16 GEMM NT:", flush=True)

        # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
        for (
            _,
            num_groups,
            max_m,
            expected_m_per_group,
            n,
            k,
        ) in enumerate_m_grouped_masked():
            # Test correctness
            for i in range(10):
                a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                    num_groups, max_m, expected_m_per_group, n, k, use_bf16=True
                )
                m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)
                for j in range(num_groups):
                    diff = calc_diff(
                        d[j, : masked_m[j].item()], ref_d[j, : masked_m[j].item()]
                    )
                    assert (
                        diff < 0.001
                    ), f"{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}"

            # Construct full cases
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k, use_bf16=True
            )

            # noinspection PyShadowingNames
            def test_func():
                m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)

            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, "bf16_gemm", suppress_kineto_output=True)
            print(
                f" > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): "
                f"{t * 1e6:4.0f} us | "
                f"{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | "
                f"{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
