
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "common",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//rtp_llm/cpp/core:copy_ops_hdrs",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/cpp/kernels:fused_copy_kernel",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/models_py/bindings/core:sampling_ops_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
