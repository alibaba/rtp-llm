
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "fuse_copy_op",
    srcs = ["FusedCopyOp.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:types_hdr",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
        ],
        "//:using_xpu": [],
        "//conditions:default": [],
    }),
    features = select({
        "//:using_xpu": ["xpu_sycl_compile"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

cc_library(
    name = "common",
    srcs = glob(
        ["*.cc"],
        exclude = [
            "CudaGraphPrefillCopy.cc",
            "FusedCopyOp.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
    ) + select({
        "//:using_cuda": [
            "CudaGraphPrefillCopy.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
        "//:using_rocm": [
            "CudaGraphPrefillCopy.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
        "//:using_xpu": [],
        "//conditions:default": [],
    }),
    hdrs = glob(["*.h"]),
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_util",
    ] + torch_deps() + select({
        "//:using_cuda": [
            ":fuse_copy_op",
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "//:using_rocm": [
            ":fuse_copy_op",
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
        ],
        "//:using_xpu": [
            ":fuse_copy_op",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
