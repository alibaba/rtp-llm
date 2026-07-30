load("//:def.bzl", "cc_test_wrapper", "copts")

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

# Keep the original label available for internal architecture overlays that
# have not yet migrated to the shorter target name.
alias(
    name = "cpu_tp_broadcaster",
    actual = ":cpu_broadcaster",
)

cc_test(
    name = "cpu_broadcaster_test",
    srcs = ["test/CpuBroadcasterTest.cc"],
    deps = [
        ":cpu_broadcaster",
        "//rtp_llm/cpp/testing:test_log_capture",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
    timeout = "short",
)

cc_test(
    name = "cpu_broadcast_shared_library_test",
    srcs = ["test/CpuBroadcastSharedLibraryTest.cc"],
    # Load the same production libraries shipped to Python. In particular,
    # libth_transformer.so contains tpSyncModelInputs and resolves its
    # execBroadcastCpu reference from librtp_compute_ops.so at load time.
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
    ],
    # libth_transformer.so has a runtime dependency on libcuda.so.1 even
    # though this test only inspects symbols. Route it to the H20 pool instead
    # of a driverless remote worker.
    exec_properties = {"gpu": "H20"},
    linkopts = ["-ldl"],
    tags = ["H20"],
    timeout = "short",
)
