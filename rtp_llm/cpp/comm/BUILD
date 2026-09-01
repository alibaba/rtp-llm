load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# Header-only collective call interface. Safe to link into every .so.
cc_library(
    name = "collective_types",
    hdrs = ["CollectiveTypes.h"],
    deps = torch_deps(),
    copts = copts(),
)

cc_library(
    name = "collective_backend_hdrs",
    hdrs = ["CollectiveBackend.h"],
    deps = [":collective_types"],
    copts = copts(),
)

cc_library(
    name = "collective_backend_pybind_hdrs",
    hdrs = ["CollectiveBackendPybind.h"],
    deps = torch_deps(),
    copts = copts(),
)

# Implementation + the `g_*_fn` callback trio.  Must live in EXACTLY ONE .so
# (librtp_compute_ops.so via th_compute_lib) so that the Python-side
# `register_comm_ops` call and the engine-side `execAllGather` call see the
# same function pointers.  Linking this into libth_transformer.so would give
# it its own unregistered copy, which crashes the engine at KV cache init.
cc_library(
    name = "collective_backend_impl",
    srcs = ["CollectiveBackend.cc"],
    deps = [
        ":collective_backend_hdrs",
        ":collective_backend_pybind_hdrs",
        "//rtp_llm/cpp/distribute:cpu_tp_broadcaster",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/cpp/comm/test:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)
