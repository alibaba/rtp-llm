load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

package(default_visibility = ["//visibility:public"])

# Pure CPU broadcaster used by tpSyncModelInputs to bypass NCCL's
# cudaDeviceSynchronize stall on the small per-step CPU tensors.
# Star topology over Unix Domain Sockets, intra-node TP only.
cc_library(
    name = "cpu_broadcaster",
    srcs = ["CpuBroadcaster.cc"],
    hdrs = ["CpuBroadcaster.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
)

cc_library(
    name = "cpu_broadcast_hdr",
    hdrs = ["CpuBroadcast.h"],
    deps = [
        "//rtp_llm/models_py/bindings/core:op_data",
    ],
    copts = copts(),
)

cc_library(
    name = "cpu_broadcast",
    srcs = ["CpuBroadcast.cc"],
    deps = [
        ":cpu_broadcast_hdr",
        ":cpu_broadcaster",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ] + torch_deps(),
    copts = copts(),
)

cc_test(
    name = "cpu_broadcaster_test",
    srcs = ["test/CpuBroadcasterTest.cc"],
    deps = [
        ":cpu_broadcaster",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
    timeout = "short",
)
