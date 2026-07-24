
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "common",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//rtp_llm/cpp/cache:cache_store_writer",
        "//rtp_llm/cpp/core:copy_ops",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
