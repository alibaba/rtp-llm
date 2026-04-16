
load("//bazel:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "fuse_copy_op",
    srcs = ["FusedCopyOp.cc"],
    deps = [
        "//rtp_llm/cpp/core:exec_ops_hdr",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/cpp/rocm:rocm_types_hdr",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

cc_library(
    name = "common",
    srcs = glob(["*.cc"], exclude = ["FusedCopyOp.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        ":fuse_copy_op",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/core:op_data",
        "//rtp_llm/cpp/core:exec_ops_hdr",
        "//rtp_llm/cpp/core:cache_store_async_writer",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/cpp/cuda/ops:cuda_impl",
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/cpp/core:exec_ctx_rocm",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
