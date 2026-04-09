load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "split_kv_cache_copy",
    srcs = ["SplitKvCacheCopy.cc"],
    hdrs = ["SplitKvCacheCopy.h"],
    copts = copts(),
    deps = torch_deps() + [
        "//rtp_llm/models_py/bindings/kernels:sm_copy_kernel",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
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
        "//rtp_llm/cpp/cuda:cuda_host_utils",
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
    ]),
    hdrs = glob([
        "*.h",
    ], exclude = [
        "SplitKvCacheCopy.h",
    ]),
    copts = cuda_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_ep_util",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache",
        "//rtp_llm/models_py/bindings/cuda/kernels:fast_topk",
        "//rtp_llm/models_py/bindings/cuda/kernels:mla_quant_kernel",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/models_py/bindings/cuda/kernels:user_buffers",
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
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":cuda_bindings_base",
        ":cuda_attn_bindings_base"
    ] + select({
        "@//:using_cuda12_9": [
            "//rtp_llm/cpp/cuda/cutlass:fp4_gemm_cu",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
