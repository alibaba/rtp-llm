load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")


cc_library(
    name = "cuda_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterOps.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = cuda_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/models_py/bindings:op_defs",
    ],
)

cc_library(
    name = "cuda_register",
    srcs = [
        "RegisterOps.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":cuda_bindings",
    ],
    visibility = ["//visibility:public"],
)
