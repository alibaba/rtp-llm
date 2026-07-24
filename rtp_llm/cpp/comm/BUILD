load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# Header-only declarations of execBroadcast / execAllReduce / execAllGather /
# execSyncCommunication and the pybind registration entry.  Safe to link into
# every .so; provides no symbols with state.
cc_library(
    name = "collective_backend_hdrs",
    hdrs = ["CollectiveBackend.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:op_data",
    ] + torch_deps(),
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
    ] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)

# Compat alias; most callers only need the header.  Anything that previously
# depended on `:collective_backend` for its side of the pybind/engine boundary
# should keep working because the impl lives in the single .so they load.
alias(
    name = "collective_backend",
    actual = ":collective_backend_hdrs",
)
