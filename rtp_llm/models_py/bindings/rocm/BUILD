load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "rocm_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterRocmOps.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = rocm_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
    ],
)

# These bindings are to be reused by ppu bindings.
cc_library(
    name = "rocm_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "rocm_attn_bindings_base",
    hdrs = [
        "RegisterAttnOpBindings.hpp"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "rocm_bindings_register",
    srcs = [
        "RegisterRocmOps.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings_base",
        ":rocm_attn_bindings_base",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
