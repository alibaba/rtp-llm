"""Standalone CUTLASS FP4 Group GEMM benchmark wrapper.

Source-integrates the vLLM CUTLASS FP4 MoE kernel (nvfp4_blockwise_moe_kernel.cu)
by adapting torch::stable::Tensor -> torch::Tensor and JIT-compiling via
torch.utils.cpp_extension.load().

The kernel is CUTLASS GemmUniversal with BlockScaledTensorOp, SM100 only.
"""
import functools
import os
import tempfile
import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def swizzle_blockscale(scale_2d, M, K_div16):
    """Swizzle block-scale tensor to CUTLASS tcgen05 layout.

    Input: [M, K_div16] float8_e4m3fn
    Output: [M_padded, K_padded] float8_e4m3fn (swizzled)
    """
    M_pad = ((M + 127) // 128) * 128
    K_pad = ((K_div16 + 3) // 4) * 4

    padded = torch.zeros(M_pad, K_pad, device=scale_2d.device, dtype=scale_2d.dtype)
    padded[:M, :K_div16] = scale_2d[:M, :K_div16]

    reshaped = padded.view(M_pad // 128, 4, 32, K_pad // 4, 4)
    swizzled = reshaped.permute(0, 3, 2, 1, 4).contiguous()
    return swizzled.view(M_pad, K_pad)



def _find_cutlass_include():
    """Find CUTLASS 4.x headers from pip packages or known paths."""
    # Try pip-installed nvidia-cutlass-dsl
    try:
        import nvidia_cutlass_dsl
        pkg_dir = os.path.dirname(nvidia_cutlass_dsl.__file__)
        candidate = os.path.join(pkg_dir, "python_packages", "cutlass", "include")
        if os.path.exists(os.path.join(candidate, "cutlass", "cutlass.h")):
            tools = os.path.join(pkg_dir, "python_packages", "cutlass", "tools", "util", "include")
            return candidate, tools if os.path.exists(tools) else candidate
    except ImportError:
        pass
    # Try cutlass package
    try:
        import cutlass
        pkg_dir = os.path.dirname(cutlass.__file__)
        for rel in ["include", "../include", "../../include"]:
            candidate = os.path.normpath(os.path.join(pkg_dir, rel))
            if os.path.exists(os.path.join(candidate, "cutlass", "cutlass.h")):
                tools = os.path.normpath(os.path.join(candidate, "..", "tools", "util", "include"))
                return candidate, tools if os.path.exists(tools) else candidate
    except ImportError:
        pass
    # Known paths
    for base in ["/dev/shm/liukan.lk/cutlass", "/usr/local/cutlass"]:
        inc = os.path.join(base, "include")
        tools = os.path.join(base, "tools", "util", "include")
        if os.path.exists(os.path.join(inc, "cutlass", "cutlass.h")):
            return inc, tools if os.path.exists(tools) else inc
    raise ImportError("CUTLASS 4.x headers not found (need cutlass/cutlass.h)")


@functools.lru_cache(maxsize=1)
def _try_load_cutlass_fp4_module():
    """JIT-compile the bundled CUTLASS FP4 kernel."""
    from torch.utils.cpp_extension import load

    # Use the bundled adapted source (same directory as this file)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cu_src = os.path.join(this_dir, "cutlass_fp4_group_mm.cu")
    if not os.path.exists(cu_src):
        raise ImportError(f"Bundled kernel source not found: {cu_src}")

    cutlass_include, cutlass_tools = _find_cutlass_include()

    build_dir = os.path.join(tempfile.gettempdir(), "cutlass_fp4_jit")
    os.makedirs(build_dir, exist_ok=True)

    module = load(
        name="cutlass_fp4_group_mm_ext",
        sources=[cu_src],
        extra_include_paths=[cutlass_include, cutlass_tools],
        extra_cuda_cflags=[
            "-DENABLE_NVFP4_SM100=1",
            "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
        ],
        build_directory=build_dir,
        verbose=False,
    )
    return module


def bench_cutlass_fp4(E, tokens_per_expert, K, N, seed, warmup, iters):
    """Benchmark CUTLASS FP4 group GEMM (vLLM/SGLang kernel).

    Returns: (avg_ms, tflops, error_str_or_None)
    """
    import time

    module = _try_load_cutlass_fp4_module()
    device = "cuda"

    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E

    total_tokens = sum(tokens_per_expert)
    group_size = 16  # FP4 block scale group size

    # Weights: [E, N, K//2] uint8 (packed FP4)
    w_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32,
                           generator=torch.Generator(device).manual_seed(seed)) * 0.1).to(torch.bfloat16)
    # Quantize to FP4 (naive: cast to float8 then pack pairs)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    # Pack FP4: take pairs of fp8 values, pack as uint8
    # For benchmark purposes, use random uint8 (kernel perf doesn't depend on values)
    w_fp4 = torch.randint(0, 256, (E, N, K // 2), device=device, dtype=torch.uint8)

    # Weight blockscale: [E, N, K//group_size] float8_e4m3fn
    b_scales = torch.ones(E, N, K // group_size, device=device, dtype=torch.float8_e4m3fn)

    # Input: [total_tokens, K//2] uint8 (packed FP4)
    a_fp4 = torch.randint(0, 256, (total_tokens, K // 2), device=device, dtype=torch.uint8)

    # Input blockscale: [sum(sf_sizes), K//group_size] float8_e4m3fn
    # sf_sizes = ceil(M_i / 128) * 128 for each expert
    sf_sizes = [((m + 127) // 128) * 128 for m in tokens_per_expert]
    total_sf = sum(sf_sizes)
    a_scales = torch.ones(total_sf, K // group_size, device=device, dtype=torch.float8_e4m3fn)

    # Alphas: [E] float32
    alphas = torch.ones(E, device=device, dtype=torch.float32)

    # Problem sizes: [E, 3] int32 — (M_i, N, K)
    problem_sizes = torch.zeros(E, 3, device=device, dtype=torch.int32)
    for i in range(E):
        problem_sizes[i, 0] = tokens_per_expert[i]
        problem_sizes[i, 1] = N
        problem_sizes[i, 2] = K

    # Expert offsets: [E] int32 — cumulative token offsets
    expert_offsets = torch.zeros(E, device=device, dtype=torch.int32)
    offset = 0
    for i in range(E):
        expert_offsets[i] = offset
        offset += tokens_per_expert[i]

    # SF offsets: [E] int32 — cumulative scale factor offsets
    sf_offsets = torch.zeros(E, device=device, dtype=torch.int32)
    sf_off = 0
    for i in range(E):
        sf_offsets[i] = sf_off
        sf_off += sf_sizes[i]

    # Output: [total_tokens, N] bfloat16
    output = torch.empty(total_tokens, N, device=device, dtype=torch.bfloat16)

    def run():
        module.cutlass_fp4_group_mm(
            output, a_fp4, w_fp4, a_scales, b_scales, alphas,
            problem_sizes, expert_offsets, sf_offsets)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    avg_ms = ((time.perf_counter() - start) / iters) * 1000

    # TFLOPS: this is a single GEMM (not full MoE with FC1+FC2)
    # To be comparable with full MoE, caller needs to handle this
    flops = total_tokens * N * K * 2  # single GEMM FLOPS
    tflops = (flops / (avg_ms / 1000)) / 1e12

    return avg_ms, tflops, None
