"""Standalone CUTLASS FP4 Group GEMM benchmark wrapper.

Source-integrates the vLLM CUTLASS FP4 MoE kernel (nvfp4_blockwise_moe_kernel.cu)
by adapting torch::stable::Tensor -> torch::Tensor and JIT-compiling via
torch.utils.cpp_extension.load().

The kernel is CUTLASS GemmUniversal with BlockScaledTensorOp, SM100 only.
"""
import functools
import os
import re
import shutil
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


def _adapt_source(src_path: str, out_dir: str) -> str:
    """Read vLLM kernel source and adapt from stable ABI to standard torch API."""
    with open(src_path, "r") as f:
        src = f.read()

    # 1. Replace stable ABI includes with standard torch
    src = src.replace('#include <torch/csrc/stable/library.h>', '#include <torch/extension.h>')
    src = src.replace('#include <torch/csrc/stable/tensor.h>', '')
    src = src.replace('#include "libtorch_stable/torch_utils.h"', '')

    # 2. Replace torch::stable::Tensor -> torch::Tensor
    src = src.replace('torch::stable::Tensor', 'torch::Tensor')

    # 3. Replace torch::stable::empty -> torch::empty
    # torch::stable::empty(size, dtype, ..., device) -> torch::empty({size}, torch::TensorOptions().dtype(dtype).device(device))
    # This is complex, let's use a simpler pattern: replace the factory calls
    src = re.sub(
        r'torch::stable::empty\((\w+),\s*torch::headeronly::ScalarType::(\w+),\s*std::nullopt,\s*(\w+)\.device\(\)\)',
        r'torch::empty({\1}, torch::TensorOptions().dtype(torch::kLong).device(\3.device()))',
        src
    )
    # Handle 2D empty: torch::stable::empty({rows, cols}, ...)
    src = re.sub(
        r'torch::stable::empty\(\{(\w+),\s*(\w+)\},\s*torch::headeronly::ScalarType::(\w+),\s*std::nullopt,\s*(\w+)\.device\(\)\)',
        r'torch::empty({\1, \2}, torch::TensorOptions().dtype(torch::kLong).device(\4.device()))',
        src
    )
    # workspace: torch::stable::empty(workspace_size, Byte, ...)
    src = re.sub(
        r'torch::stable::empty\(workspace_size,\s*torch::headeronly::ScalarType::Byte,\s*std::nullopt,\s*(\w+)\.device\(\)\)',
        r'torch::empty({static_cast<long>(workspace_size)}, torch::TensorOptions().dtype(torch::kByte).device(\\1.device()))',
        src
    )

    # 4. Replace scalar type checks
    src = src.replace('torch::headeronly::ScalarType::BFloat16', 'torch::kBFloat16')
    src = src.replace('torch::headeronly::ScalarType::Half', 'torch::kHalf')
    src = src.replace('torch::headeronly::ScalarType::Float', 'torch::kFloat32')
    src = src.replace('torch::headeronly::ScalarType::Int', 'torch::kInt32')
    src = src.replace('torch::headeronly::ScalarType::Long', 'torch::kLong')
    src = src.replace('torch::headeronly::ScalarType::Byte', 'torch::kByte')

    # 5. Replace STD_TORCH_CHECK -> TORCH_CHECK
    src = src.replace('STD_TORCH_CHECK_NOT_IMPLEMENTED', 'TORCH_CHECK')
    src = src.replace('STD_TORCH_CHECK', 'TORCH_CHECK')

    # 6. Replace get_current_cuda_stream
    src = src.replace(
        'get_current_cuda_stream(a_tensors.get_device_index())',
        'at::cuda::getCurrentCUDAStream(a_tensors.get_device()).stream()'
    )
    src = src.replace(
        'get_current_cuda_stream(a.get_device_index())',
        'at::cuda::getCurrentCUDAStream(a.get_device()).stream()'
    )

    # 7. Replace get_device_index() -> get_device()
    src = src.replace('.get_device_index()', '.get_device()')

    # 8. Replace STABLE_TORCH_LIBRARY_IMPL with pybind11 module
    # Remove the library registration block
    src = re.sub(
        r'STABLE_TORCH_LIBRARY_IMPL\(_C, CUDA, m\)\s*\{[^}]*\}',
        '',
        src
    )

    # 9. Replace FLOAT4_E2M1X2 and SF_DTYPE checks (these macros may not exist)
    # Remove CHECK_INPUT lines that use unknown macros
    src = re.sub(r'#define CHECK_INPUT.*\n', '', src)
    src = re.sub(r'#define CHECK_TYPE.*\n', '', src)
    src = re.sub(r'CHECK_INPUT\([^)]*\);\n', '', src)

    # 10. Add ATen CUDA stream include
    src = '#include <ATen/cuda/CUDAContext.h>\n' + src

    # 11. Add get_sm_version_num inline implementation
    src = src.replace(
        '#include "cutlass_extensions/common.hpp"',
        '''// Inline SM version query (replaces cutlass_extensions/common.hpp)
inline int32_t get_sm_version_num() {
    int device_id;
    cudaGetDevice(&device_id);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    return major * 10 + minor;
}
'''
    )

    # 12. Add pybind11 module at the end
    src += '''
// pybind11 module
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_fp4_group_mm", &cutlass_fp4_group_mm,
          "CUTLASS FP4 Group GEMM for MoE (SM100)",
          py::arg("output"), py::arg("a"), py::arg("b"),
          py::arg("a_blockscale"), py::arg("b_blockscales"),
          py::arg("alphas"), py::arg("problem_sizes"),
          py::arg("expert_offsets"), py::arg("sf_offsets"));
}
'''

    # Write adapted source
    out_path = os.path.join(out_dir, "cutlass_fp4_group_mm.cu")
    with open(out_path, "w") as f:
        f.write(src)
    return out_path


@functools.lru_cache(maxsize=1)
def _try_load_cutlass_fp4_module():
    """JIT-compile the adapted CUTLASS FP4 kernel."""
    from torch.utils.cpp_extension import load

    vllm_src = "/dev/shm/liukan.lk/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
    cutlass_include = "/dev/shm/liukan.lk/cutlass/include"
    cutlass_tools = "/dev/shm/liukan.lk/cutlass/tools/util/include"

    if not os.path.exists(vllm_src):
        raise ImportError(f"vLLM kernel source not found: {vllm_src}")
    if not os.path.exists(cutlass_include):
        raise ImportError(f"CUTLASS headers not found: {cutlass_include}")

    # Create temp dir for adapted source
    build_dir = os.path.join(tempfile.gettempdir(), "cutlass_fp4_jit")
    os.makedirs(build_dir, exist_ok=True)

    adapted_src = _adapt_source(vllm_src, build_dir)

    module = load(
        name="cutlass_fp4_group_mm_ext",
        sources=[adapted_src],
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
