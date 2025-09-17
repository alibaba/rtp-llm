
load("//bazel:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "common",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/cpp/kernels:kernels_cu",
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "//:using_rocm": [
            "//rtp_llm/cpp/kernels:kernels_cu",
            "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
