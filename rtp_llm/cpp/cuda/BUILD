load("//:def.bzl", "rpm_library", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps", "fa_deps", "trt_plugins")
package(default_visibility=["//rtp_llm:__subpackages__"])

trt_plugins()

cc_library(
    name = "cublas",
    srcs = [
        "cublas/cublasAlgoMap.cc",
        "cublas/cublasMMWrapper.cc",
    ],
    hdrs = [
        "cublas/cublasAlgoMap.h",
        "cublas/cublasMMWrapper.h",
        "cublas/cublas.h",
    ],
    deps = [
        ":cuda_utils",
        "//rtp_llm/cpp/core:allocator",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cublasLt",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

fa_deps()

cc_library(
    name = "cufmha",
    srcs = [
        "cufmha/cufmha.cc",
        ":fa",
    ],
    hdrs = [
        "cufmha/cufmha.h",
        "cufmha/fmha_profiling_interface.h"
    ],
    deps = [
        ":fa_hdrs",
        ":cuda_utils",
        "//rtp_llm/cpp/core:types",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_header",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_header",
        ":memory_utils",
        "//rtp_llm/cpp/kernels:kernels",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/contextFusedMultiHeadAttentionSm70:trt_sm70_fmha_impl",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuggemm",
    srcs = [
        "cuggemm/cuggemm.cc",
    ],
    hdrs = [
        "cuggemm/cuggemm.h",
    ],
    deps = [
        ":cuda_utils",
        "//rtp_llm/cpp/core:types",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":trt_plugins",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "custom_ar",
    srcs = [
        "custom_ar/custom_ar_comm.cc",
    ],
    hdrs = [
        "custom_ar/custom_ar_comm.h"
    ],
    deps = [
        ":nccl_util",
        ":cuda_utils",
        ":memory_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "comm_buffer",
    srcs = [
        "comm_buffer/comm_buffer.cc",
    ],
    hdrs = [
        "comm_buffer/comm_buffer.h"
    ],
    deps = [
        ":nccl_util",
        ":custom_ar",
        ":cuda_utils",
        ":memory_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_util",
    srcs = [
        "nccl/nccl_utils_torch.cc",
        "nccl/nccl_utils.cc",
    ],
    hdrs = [
        "nccl/nccl_utils_torch.h",
        "nccl/nccl_utils.h",
    ],
    deps = torch_deps() + select({
        "@//:using_cuda": [":cuda_utils",
                           "@local_config_cuda//cuda:cuda_headers",
                           "@local_config_cuda//cuda:cudart",],
        "@//:using_rocm": ["//rtp_llm/cpp/rocm:rocm_utils"],
        "//conditions:default": [],
    }),
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvtx",
    srcs = [
        "nvtx/nvtx_utils.cc",
        "nvtx/kernel_profiler.cc"
    ],
    hdrs = [
        "nvtx/nvtx_utils.h",
        "nvtx/kernel_profiler.h"
    ],
    deps = [
        "//rtp_llm/cpp:metrics",
    ] + select({
        "@//:using_cuda": [":cuda_utils",
                           "@local_config_cuda//cuda:cuda_headers",
                           "@local_config_cuda//cuda:cudart",],
        "@//:using_rocm": ["//rtp_llm/cpp/rocm:rocm_utils"],
        "//conditions:default": [],
    }),
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_utils",
    srcs = [
        "memory_utils.cu",
    ],
    hdrs = [
        "memory_utils.h",

    ],
    deps = [
        ":cuda_utils",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/core:allocator",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils_cu",
    srcs = [
        "cuda_fp8_utils.cu"
    ],
    hdrs = [
        "cuda_fp8_utils.h",
        "cuda_type_utils.cuh",
        "cuda_bf16_fallbacks.cuh",
        "reduce_kernel_utils.cuh"
    ],
    deps = [
        ":cuda_utils_base",
        "//rtp_llm/cpp:utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_fmha_utils",
    hdrs = [
        "cuda_fmha_utils.h",
    ],
    deps = [
        ":cuda_utils",
        "//rtp_llm/cpp:utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils_base",
    srcs = [
        "cuda_utils.cc",
        "launch_utils.cc",
    ],
    hdrs = [
        "launch_utils.h",
        "cuda_utils.h",
        "Dispatch.h",
        "trt_utils.h",
        "quantize_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/core:types",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils",
    deps = [
        ":cuda_utils_cu",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/core:types",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:nvml",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = [
    ],
    hdrs = [
        "Dispatch.h"],
    deps = [
        ":nccl_util",
        ":nvtx",
        ":cublas",
        ":cufmha",
        "cuggemm",
        ":cuda_utils",
        ":cuda_fmha_utils",
        ":custom_ar",
        ":comm_buffer",
        ":memory_utils",
        "//rtp_llm/cpp/kernels:kernels",
        "//rtp_llm/cpp:utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_cuda",
    srcs = [
        "allocator_cuda.cc"
    ],
    hdrs = [
        "allocator_cuda.h"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/core:allocator",
        ":cuda_utils",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_torch",
    srcs = [
        "allocator_torch.cc"
    ],
    hdrs = [
        "allocator_torch.h"
    ],
    copts = copts(),
    deps = torch_deps() + [
        ":allocator_cuda",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_utils",
    srcs = [
        "tensor_utils.cc"
    ],
    hdrs = [
        "tensor_utils.h"
    ],
    copts = copts(),
    deps = torch_deps(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "expert_attention_util",
    srcs = [
        "ExpertAttentionUtil.cc"
    ],
    hdrs = [
        "ExpertAttentionUtil.h"
    ],
    copts = copts(),
    deps = torch_deps() + [
        ":cuda_utils",
        "//rtp_llm/cpp/core:allocator",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils_rocm",
    srcs = [
        "launch_utils.cc",
    ],
    hdrs = [
        "launch_utils.h",
        "cuda_utils.h",
        "Dispatch.h",
        "memory_utils.h",
        "cuda_fp8_utils.h",
        "cuda_type_utils.cuh",
        "cuda_bf16_fallbacks.cuh",
        "reduce_kernel_utils.cuh"
    ],
    deps = [
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/rocm:rocm_utils",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
