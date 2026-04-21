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
        "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
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

cc_library(
    name = "cutlass_headers",
    deps = [
        "@cutlass//:cutlass",
        "@cutlass//:cutlass_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_interface",
    hdrs = [
        "cutlass_kernels/weight_only_quant_op.h",
        "cutlass_kernels/gemm_configs.h",
    ],
    visibility = ["//visibility:public"],
)

# Stub targets for removed cutlass MOE code (now replaced by Triton fused_moe).
# Kept as empty shared libraries so that internal arch_select.bzl copy_so() references
# don't break the build.
cc_library(
    name = "moe_sm90_stub",
    srcs = [],
    visibility = ["//visibility:private"],
)

cc_shared_library(
    name = "moe_sm90",
    roots = [":moe_sm90_stub"],
    visibility = ["//visibility:public"],
)
