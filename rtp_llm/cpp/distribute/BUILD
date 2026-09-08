load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

package(default_visibility = ["//visibility:public"])

# Pure CPU broadcaster used by tpSyncModelInputs to bypass NCCL's
# cudaDeviceSynchronize stall on the small per-step CPU tensors.
# Star topology over Unix Domain Sockets, intra-node TP only.
cc_library(
    name = "cpu_tp_broadcaster",
    srcs = ["CpuTpBroadcaster.cc"],
    hdrs = ["CpuTpBroadcaster.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
)

cc_test(
    name = "cpu_tp_broadcaster_test",
    srcs = ["test/CpuTpBroadcasterTest.cc"],
    deps = [
        ":cpu_tp_broadcaster",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
    timeout = "short",
)
