load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load(":moe.bzl", "gen_moe_kernels")
load("@//bazel:arch_select.bzl", "torch_deps")

template_header = """
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/moe_gemm/moe_kernels.inl"

namespace tensorrt_llm
{
"""
template = """
template class MoeGemmRunner<{0}, {1}, {3}, {2}>;
template class CutlassMoeFCRunner<{0}, {1}, {3}>;
"""
template_tail = """
}
"""
T = [
    ('float', 'float', 'float', 'cutlass::WeightOnlyQuantOp::UNDEFINED'),
    ('half', 'half', 'half', 'cutlass::WeightOnlyQuantOp::UNDEFINED'),
    ('half', 'cutlass::uint4b_t', 'half', 'cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS'),
    ('half', 'cutlass::uint4b_t', 'half', 'cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY'),
    ('half', 'uint8_t', 'half', 'cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS'),
    ('half', 'uint8_t', 'half', 'cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY'),
    ('__nv_bfloat16', '__nv_bfloat16', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::UNDEFINED'),
    ('__nv_bfloat16', 'cutlass::uint4b_t', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS'),
    ('__nv_bfloat16', 'cutlass::uint4b_t', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY'),
    ('__nv_bfloat16', 'uint8_t', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS'),
    ('__nv_bfloat16', 'uint8_t', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY'),
    # ('__nv_fp8_e4m3', '__nv_fp8_e4m3', 'half', 'cutlass::WeightOnlyQuantOp::UNDEFINED'),
    # ('__nv_fp8_e4m3', '__nv_fp8_e4m3', '__nv_bfloat16', 'cutlass::WeightOnlyQuantOp::UNDEFINED'),
]

gen_cpp_code("moe_runner", [T],
             template_header, template, template_tail, element_per_file=1, suffix=".cu")

gen_moe_kernels()

sm90_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_90', '--cuda-gpu-arch=sm_90',
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
]

sm100_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_100', '--cuda-gpu-arch=sm_100',
    '--cuda-include-ptx=sm_100a', '--cuda-gpu-arch=sm_100a',
]

cc_library(
    name = "moe_cu",
    srcs = glob([
        "cutlass_kernels/moe_gemm/*.cu",
    ]) + [
        ":moe_runner",
    ":moe_inst_sm80",
    ],
    hdrs = glob([
        "cutlass_kernels/moe_gemm/*.h",
        "cutlass_kernels/moe_gemm/**/*.h",
        "cutlass_kernels/moe_gemm/**/*.inl",
    ]),
    include_prefix = "src",
    deps = [
        ":moe_cu_sm90",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts() + [
        '-DCOMPILE_HOPPER_TMA_GEMMS',
    ],
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_shared_library(
    name = "moe",
    roots = [":moe_cu"],
    preloaded_deps = [
        ":moe_cu_sm90",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "moe_cu_sm90",
    srcs = [
        ":moe_inst_sm90",
    ],
    hdrs = glob([
        "cutlass_kernels/moe_gemm/*.h",
        "cutlass_kernels/moe_gemm/**/*.h",
        "cutlass_kernels/moe_gemm/**/*.inl",
    ]) + [
        "cutlass_kernels/cutlass_type_conversion.h",
    ],
    include_prefix = "src",
    deps = [
    "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    copts = sm90_cuda_copts + [
        '-DCOMPILE_HOPPER_TMA_GEMMS',
    ],
    alwayslink = True,
)

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

cc_shared_library(
    name = "moe_sm90",
    roots = [":moe_cu_sm90"],
    static_deps = [
        "@local_config_cuda//cuda:cuda_driver",
    ],
    preloaded_deps = [
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
    ],
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
    name = "cutlass_kernels_common",
    srcs = glob([
        "cutlass_kernels/*.cc",
    ]),
    hdrs = glob([
        "cutlass_kernels/*.h",
    ]),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/cutlass/cutlass_extensions:cutlass_extensions",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "@cutlass//:cutlass",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "cutlass_kernels_impl",
    srcs = [
        ":moe",
        ":moe_sm90",
    ],
    deps = [
        ":cutlass_kernels_common",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_interface",
    hdrs = [
        "interface.h",
        "cutlass_kernels/moe_gemm/moe_kernels.h",
        "cutlass_kernels/moe_gemm/moe_kernels.inl",
        "cutlass_kernels/moe_gemm/moe_fp8_kernels.h",
        "cutlass_kernels/moe_gemm/moe_gemm_kernels.h",
        "cutlass_kernels/weight_only_quant_op.h",
        "cutlass_kernels/gemm_configs.h",
        "cutlass_kernels/cutlass_heuristic.h",
        "cutlass_kernels/gemm_lut_utils.h",
        "cutlass_kernels/gemm_lut.h",
    ],
    visibility = ["//visibility:public"],
)
