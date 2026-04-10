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
    # Try deep_gemm package (SGLang installs CUTLASS headers into deep_gemm/include/)
    try:
        import deep_gemm
        pkg_dir = os.path.dirname(deep_gemm.__file__)
        candidate = os.path.join(pkg_dir, "include")
        if os.path.exists(os.path.join(candidate, "cutlass", "cutlass.h")):
            tools = os.path.join(candidate, "..", "tools", "util", "include")
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
    # Search sys.path for CUTLASS headers (handles Bazel runfiles)
    import sys
    for p in sys.path:
        for sub in [
            "deep_gemm/include",
            "nvidia_cutlass_dsl/python_packages/cutlass/include",
            "nvidia_cutlass_dsl/include",
        ]:
            candidate = os.path.join(p, sub)
            if os.path.exists(os.path.join(candidate, "cutlass", "cutlass.h")):
                tools = os.path.join(os.path.dirname(candidate), "tools", "util", "include")
                return candidate, tools if os.path.exists(tools) else candidate
    # Search RUNFILES_DIR for cutlass headers
    runfiles = os.environ.get("RUNFILES_DIR", "")
    if runfiles:
        for root, dirs, _ in os.walk(runfiles):
            if "cutlass" in dirs:
                candidate = os.path.join(root, "cutlass")
                if os.path.exists(os.path.join(root, "cutlass", "cutlass.h")):
                    # root is the include dir
                    tools = os.path.join(os.path.dirname(root), "tools", "util", "include")
                    return root, tools if os.path.exists(tools) else root
            if root.count(os.sep) - runfiles.count(os.sep) > 6:
                break  # don't recurse too deep
    # Known paths
    for base in ["/dev/shm/liukan.lk/cutlass", "/usr/local/cutlass"]:
        inc = os.path.join(base, "include")
        tools = os.path.join(base, "tools", "util", "include")
        if os.path.exists(os.path.join(inc, "cutlass", "cutlass.h")):
            return inc, tools if os.path.exists(tools) else inc
    raise ImportError(
        f"CUTLASS 4.x headers not found (need cutlass/cutlass.h). "
        f"sys.path has {len(sys.path)} entries, RUNFILES_DIR={runfiles[:100]}"
    )


def _find_cu_source():
    """Find the bundled .cu source, handling Bazel runfiles."""
    # Try same directory as this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(this_dir, "cutlass_fp4_group_mm.cu")
    if os.path.exists(candidate):
        return candidate
    # Try Bazel runfiles
    runfiles_dir = os.environ.get("RUNFILES_DIR", "")
    if runfiles_dir:
        for pattern in [
            "rtp_llm/rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test/cutlass_fp4_group_mm.cu",
            "rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test/cutlass_fp4_group_mm.cu",
        ]:
            candidate = os.path.join(runfiles_dir, pattern)
            if os.path.exists(candidate):
                return candidate
    # Search recursively from runfiles
    if runfiles_dir:
        for root, _, files in os.walk(runfiles_dir):
            if "cutlass_fp4_group_mm.cu" in files:
                return os.path.join(root, "cutlass_fp4_group_mm.cu")
    raise ImportError(
        f"cutlass_fp4_group_mm.cu not found. "
        f"Searched: {this_dir}, RUNFILES_DIR={runfiles_dir}"
    )


def _ensure_ninja():
    """Ensure ninja binary is in PATH (required by torch cpp_extension)."""
    import shutil
    import subprocess
    import sys

    if shutil.which("ninja"):
        return

    # ninja pip package exposes BIN_DIR with the binary location
    try:
        import ninja
        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir and os.path.isfile(os.path.join(bin_dir, "ninja")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return
        # Fallback: search common locations relative to package
        pkg_dir = os.path.dirname(os.path.abspath(ninja.__file__))
        for sub in ["data/bin", ".", "../../../bin", "../../bin", "../bin"]:
            candidate = os.path.normpath(os.path.join(pkg_dir, sub))
            if os.path.isfile(os.path.join(candidate, "ninja")):
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                return
        # Last resort: find ninja binary anywhere under the package tree
        site_pkg = os.path.dirname(pkg_dir)
        for root, _, files in os.walk(site_pkg):
            if "ninja" in files and os.access(os.path.join(root, "ninja"), os.X_OK):
                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                return
    except ImportError:
        pass

    # ninja not found — try pip install
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ninja", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass


@functools.lru_cache(maxsize=1)
def _try_load_cutlass_fp4_module():
    """JIT-compile the bundled CUTLASS FP4 kernel."""
    _ensure_ninja()
    from torch.utils.cpp_extension import load

    cu_src = _find_cu_source()

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


def _prepare_fp4_gemm_args(E, tokens_per_expert, out_dim, in_dim, seed_offset, device="cuda"):
    """Prepare FP4 GEMM arguments for one matmul stage.

    Args:
        E: number of experts
        tokens_per_expert: list of token counts per expert
        out_dim: output dimension (2N for FC1, K for FC2)
        in_dim: input dimension (K for FC1, N for FC2)
        seed_offset: seed offset for weight generation
    Returns: (w_fp4, b_scales, problem_sizes, expert_offsets, sf_offsets, sf_sizes)
    """
    total_tokens = sum(tokens_per_expert)
    group_size = 16

    w_fp4 = torch.randint(0, 256, (E, out_dim, in_dim // 2), device=device, dtype=torch.uint8)
    b_scales = torch.ones(E, out_dim, in_dim // group_size, device=device, dtype=torch.float8_e4m3fn)

    problem_sizes = torch.zeros(E, 3, device=device, dtype=torch.int32)
    for i in range(E):
        problem_sizes[i, 0] = tokens_per_expert[i]
        problem_sizes[i, 1] = out_dim
        problem_sizes[i, 2] = in_dim

    expert_offsets = torch.zeros(E, device=device, dtype=torch.int32)
    offset = 0
    for i in range(E):
        expert_offsets[i] = offset
        offset += tokens_per_expert[i]

    sf_sizes = [((m + 127) // 128) * 128 for m in tokens_per_expert]
    total_sf = sum(sf_sizes)
    sf_offsets = torch.zeros(E, device=device, dtype=torch.int32)
    sf_off = 0
    for i in range(E):
        sf_offsets[i] = sf_off
        sf_off += sf_sizes[i]

    return w_fp4, b_scales, problem_sizes, expert_offsets, sf_offsets, sf_sizes


def bench_cutlass_fp4(E, tokens_per_expert, K, N, seed, warmup, iters):
    """Benchmark CUTLASS FP4 Full MoE: FC1(K→2N) + SiLU + FC2(N→K).

    Runs TWO group GEMMs per iteration to match other Full MoE implementations.
    Returns: (avg_ms, tflops, error_str_or_None)
    """
    import time
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    module = _try_load_cutlass_fp4_module()
    device = "cuda"

    if isinstance(tokens_per_expert, int):
        tokens_per_expert = [tokens_per_expert] * E

    total_tokens = sum(tokens_per_expert)
    group_size = 16

    # --- FC1: [total_tokens, K] → [total_tokens, 2N] ---
    w1_fp4, w1_scales, ps1, eo1, sfo1, sf_sizes1 = \
        _prepare_fp4_gemm_args(E, tokens_per_expert, 2 * N, K, seed, device)
    a1_fp4 = torch.randint(0, 256, (total_tokens, K // 2), device=device, dtype=torch.uint8)
    total_sf1 = sum(sf_sizes1)
    a1_scales = torch.ones(total_sf1, K // group_size, device=device, dtype=torch.float8_e4m3fn)
    alphas1 = torch.ones(E, device=device, dtype=torch.float32)
    fc1_out = torch.empty(total_tokens, 2 * N, device=device, dtype=torch.bfloat16)

    # --- FC2: [total_tokens, N] → [total_tokens, K] ---
    w2_fp4, w2_scales, ps2, eo2, sfo2, sf_sizes2 = \
        _prepare_fp4_gemm_args(E, tokens_per_expert, K, N, seed + 1, device)
    a2_fp4 = torch.randint(0, 256, (total_tokens, N // 2), device=device, dtype=torch.uint8)
    total_sf2 = sum(sf_sizes2)
    a2_scales = torch.ones(total_sf2, N // group_size, device=device, dtype=torch.float8_e4m3fn)
    alphas2 = torch.ones(E, device=device, dtype=torch.float32)
    fc2_out = torch.empty(total_tokens, K, device=device, dtype=torch.bfloat16)

    # SiLU intermediate
    act_out = torch.empty(total_tokens, N, device=device, dtype=torch.bfloat16)

    def run():
        # FC1: [M, K] x [2N, K]^T → [M, 2N]
        module.cutlass_fp4_group_mm(
            fc1_out, a1_fp4, w1_fp4, a1_scales, w1_scales, alphas1,
            ps1, eo1, sfo1)
        # SiLU activation: [M, 2N] → [M, N]
        silu_and_mul(act_out, fc1_out)
        # FC2: [M, N] x [K, N]^T → [M, K]
        # Note: FC2 input is bfloat16 from SiLU, we use pre-generated FP4
        # data for kernel-level benchmark (perf doesn't depend on values)
        module.cutlass_fp4_group_mm(
            fc2_out, a2_fp4, w2_fp4, a2_scales, w2_scales, alphas2,
            ps2, eo2, sfo2)

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

    # Full MoE TFLOPS: FC1(M*2N*K*2) + FC2(M*K*N*2)
    flops = total_tokens * (2 * N * K * 2 + K * N * 2)
    tflops = (flops / (avg_ms / 1000)) / 1e12

    return avg_ms, tflops, None
