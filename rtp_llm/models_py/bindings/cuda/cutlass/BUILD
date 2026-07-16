load("//:def.bzl", "copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("@//bazel:arch_select.bzl", "torch_deps")

sm100_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_100', '--cuda-gpu-arch=sm_100',
    '--cuda-include-ptx=sm_100a', '--cuda-gpu-arch=sm_100a',
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
    ],
    alwayslink = True,
    visibility = ["//visibility:public"],
)
