load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility=["//rtp_llm:__subpackages__"])

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
            "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
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
    copts = copts(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

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
        ":cuda_host_utils",
        "//rtp_llm/models_py/bindings/core:types",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_tensor_ops",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/contextFusedMultiHeadAttentionSm70:trt_sm70_fmha_impl",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache"
    ],
    # for fused_multihead_attention_utils.h: alpha = reinterpret_cast<uint32_t const&>(h2)
    copts = copts() + ["-fno-strict-aliasing"],
)

cc_library(
    name = "trt_utils",
    hdrs = [
        "trt_utils.h",
    ],
    deps = [
        ":cuda_host_utils",
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
        ":trt_v2_attn",
        ":cuda_host_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# --- Bindings targets (original models_py/bindings/cuda targets) ---

cc_library(
    name = "split_kv_cache_copy",
    srcs = ["SplitKvCacheCopy.cc"],
    hdrs = ["SplitKvCacheCopy.h"],
    copts = copts(),
    deps = torch_deps() + [
        "//rtp_llm/models_py/bindings/common/kernels:sm_copy_kernel",
        ":cuda_host_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "no_block_copy",
    srcs = ["NoBlockCopy.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "//rtp_llm/cpp/utils:core_utils",
        ":split_kv_cache_copy",
        ":cuda_host_utils",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "cuda_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterCudaOps.cc",
        "SplitKvCacheCopy.cc",
        "NoBlockCopy.cc",
        "cuda_host_utils.cc",
        "launch_utils.cc",
    ]),
    hdrs = glob([
        "*.h",
    ], exclude = [
        "SplitKvCacheCopy.h",
        "cuda_host_utils.h",
        "launch_utils.h",
        "cuda_fp8_utils.h",
        "trt_utils.h",
    ]),
    copts = cuda_copts(),
    deps = torch_deps() + [
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_ep_util",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache",
        "//rtp_llm/models_py/bindings/cuda/kernels:fast_topk",
        "//rtp_llm/models_py/bindings/cuda/kernels:mla_quant_kernel",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/models_py/bindings/cuda/kernels:user_buffers",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + select({
        "@//:using_cuda12_9": [
            "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp4_quant",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "cuda_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        "//rtp_llm/models_py/bindings/cuda/kernels:mla_quant_kernel",
        ":cuda_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "cuda_attn_bindings_base",
    hdrs = [
        "RegisterAttnOpBindings.hpp"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":cuda_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "cuda_bindings_register",
    srcs = [
        "RegisterCudaOps.cc",
    ],
    copts = copts() + ["-fno-strict-aliasing"],
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":cuda_bindings_base",
        ":cuda_attn_bindings_base"
    ] + select({
        "@//:using_cuda12_9": [
            "//rtp_llm/models_py/bindings/cuda/cutlass:fp4_gemm_cu",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
