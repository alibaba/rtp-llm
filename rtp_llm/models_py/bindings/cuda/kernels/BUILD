load("//:def.bzl", "copts", "cuda_copts", "any_cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//rtp_llm:__subpackages__"])

sm100_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_100', '--cuda-gpu-arch=sm_100',
    '--cuda-include-ptx=sm_100a', '--cuda-gpu-arch=sm_100a',
]

cuda_deps = [
    "//rtp_llm/cpp/cuda:cuda_host_utils",
    ":cuda_utils_common",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
]

cc_library(
    name = "cuda_utils_common",
    hdrs = [
        "vec_dtypes.cuh",
        "util.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_fused_qk_rmsnorm",
    srcs = ["fused_qk_rmsnorm.cu"],
    hdrs = ["fused_qk_rmsnorm.h"],
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_no_aux_tc",
    srcs = ["no_aux_tc_kernels.cu"],
    hdrs = ["no_aux_tc_kernels.h"],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:launch_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = cuda_copts(),
)

cc_library(
    name = "scaled_fp8_quant",
    srcs = ["scaled_fp8_quant.cu"],
    hdrs = [
        "scaled_fp8_quant.h",
        "scaled_fp8_quant_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "scaled_fp4_quant",
    srcs = ["scaled_fp4_quant.cu"],
    hdrs = ["scaled_fp4_quant.h"],
    deps = [
        "//rtp_llm/cpp/cuda/cutlass:fp4_gemm_cu",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = sm100_cuda_copts + [
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_quantization",
    srcs = [
        "per_token_group_quant_8bit.cu",
        "per_token_group_quant_8bit_v2.cu",
    ],
    hdrs = [
        "per_token_group_quant_8bit.h",
        "per_token_group_quant_8bit_v2.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
)

cc_library(
    name = "fast_topk",
    srcs = ["fast_topk.cu"],
    hdrs = ["fast_topk.h"],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mla_quant_kernel",
    srcs = ["mla_quant_kernel.cu"],
    hdrs = ["mla_quant_kernel.h"],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_cu",
    deps = [
        ":cuda_no_aux_tc",
        ":cuda_quantization",
        ":cuda_fused_qk_rmsnorm",
        ":mla_quant_kernel",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "user_buffers",
    srcs = ["user_buffer/user_buffers.cu"],
    hdrs = ["user_buffer/user_buffers.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_mla",
    srcs = glob(["mla_kernels/*.cu"]),
    hdrs = glob([
        "mla_kernels/*.h",
        "mla_kernels/*.cuh",
    ]),
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
