load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cuda_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterCudaOps.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = cuda_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/kernels:scaled_fp8_quant",
        "//rtp_llm/cpp/kernels:kernels_ep_util",
        "//rtp_llm/cpp/kernels:kernels_kv_cache",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
    ],
)

cc_library(
    name = "cuda_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
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
        "//rtp_llm/cpp/cuda/cutlass:fp8_group_cu",
        ":cuda_bindings_base",
        ":cuda_attn_bindings_base"
    ] + select({
        "@//:using_cuda12_arm": [
            "//rtp_llm/cpp/cuda/cutlass:fp4_gemm_cu",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
