load("//:def.bzl", "copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("@arch_config//:arch_select.bzl", "torch_deps")

sm100_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_100', '--cuda-gpu-arch=sm_100',
    '--cuda-include-ptx=sm_100a', '--cuda-gpu-arch=sm_100a',
    '--cuda-include-ptx=sm_120', '--cuda-gpu-arch=sm_120',
    '--cuda-include-ptx=sm_120a', '--cuda-gpu-arch=sm_120a',
]

cc_library(
    name = "fp4_gemm_cu",
    srcs = glob([
        "cutlass_kernels/fp4_gemm/*.cu",
    ]),
    hdrs = glob([
        "cutlass_kernels/fp4_gemm/*.h",
        "cutlass_kernels/fp4_gemm/*.cuh",
    ]),
    deps = [
        "@cutlass4.0//:cutlass",
        "@cutlass4.0//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_driver",
    ] + torch_deps(),
    copts = sm100_cuda_copts + [
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
        "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED",
    ],
    alwayslink = True,
    visibility = ["//visibility:public"],
)

# FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell). Ported from vLLM
# (Apache 2.0). Standalone target — no cross-deps with fp4_gemm_cu.
cc_library(
    name = "fp8_blockwise_sm120_cu",
    srcs = glob([
        "cutlass_kernels/fp8_blockwise_sm120/*.cu",
    ]),
    hdrs = glob([
        "cutlass_kernels/fp8_blockwise_sm120/*.h",
    ]),
    deps = [
        "@cutlass4.0//:cutlass",
        "@cutlass4.0//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_driver",
    ] + torch_deps(),
    copts = sm100_cuda_copts + [
        "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED",
    ],
    alwayslink = True,
    visibility = ["//visibility:public"],
)
