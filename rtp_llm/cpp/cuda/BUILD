load("//:def.bzl", "rpm_library", "copts", "cuda_copts", "host_copts")
load("//bazel:arch_select.bzl", "torch_deps", "fa_deps", "trt_plugins")
package(default_visibility=["//rtp_llm:__subpackages__"])

trt_plugins()

cc_library(
    name = "cuda_host_utils",
    srcs = [
        "cuda_host_utils.cc",
    ],
    hdrs = [
        "cuda_host_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/utils:stack_trace",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:static_config",
        "@local_config_cuda//cuda:nvml",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils_cu",
    hdrs = [
        "cuda_fp8_utils.h",
        "cuda_type_utils.cuh",
        "cuda_bf16_fallbacks.cuh",
        "reduce_kernel_utils.cuh"
    ] + select({
        "@//:using_rocm": [
            "//rtp_llm/cpp/rocm:rocm_types_hdr",
        ],
        "//conditions:default": [],
    }),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "launch_utils",
    srcs = [
        "launch_utils.cc",
    ],
    hdrs = [
        "launch_utils.h",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = host_copts(),
    visibility = ["//visibility:public"],
)

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
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:allocator",
        "@havenask//aios/autil:scope",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cublasLt",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

fa_deps()

cc_library(
    name = "trt_v2_attn",
    srcs = [
        "cufmha/TrtV2FmhaRunner.cc",
        "cufmha/TRTAttn.cc",
    ],
    hdrs = [
        "cufmha/TrtV2FmhaRunner.h",
        "cufmha/TRTAttn.h",
    ],
    deps = [
        ":fa_hdrs",
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/kernels:kernels_tensor_ops",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/contextFusedMultiHeadAttentionSm70:trt_sm70_fmha_impl",
        # still need to include devices_base for struct OpData
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/kernels:kernels_kv_cache"
    ],
    # for fused_multihead_attention_utils.h: alpha = reinterpret_cast<uint32_t const&>(h2)
    copts = copts() + ["-fno-strict-aliasing"],
)

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
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:special_tokens",
        "//rtp_llm/cpp/kernels:kernels_tensor_ops",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_header",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
        "//rtp_llm/cpp/devices:devices_base",
        ":trt_v2_attn"
    ],
    copts = copts(),
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
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":trt_plugins",
    ],
    copts = copts(),
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
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "//rtp_llm/cpp/config:config_modules",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
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
        ":cuda_host_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
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
        "@//:using_cuda": [
            ":cuda_host_utils",
            "//rtp_llm/cpp/utils:stack_trace",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/rocm:rocm_host_utils",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
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
        ":cuda_utils_cu",
        ":cuda_host_utils",
        ":launch_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/core:types",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)



cc_library(
    name = "cuda_fmha_utils",
    hdrs = [
        "cuda_fmha_utils.h",
    ],
    deps = [
        ":cuda_host_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "quantize_utils",
    hdrs = [
        "quantize_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        ":trt_plugins",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "trt_utils",
    hdrs = [
        "trt_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = [
    ],
    deps = [
        ":nccl_util",
        ":cublas",
        ":cufmha",
        ":cuggemm",
        ":cuda_host_utils",
        ":cuda_fmha_utils",
        ":custom_ar",
        ":comm_buffer",
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ],
    copts = copts(),
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
        ":cuda_host_utils",
    ],
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
    visibility = ["//visibility:public"],
)
