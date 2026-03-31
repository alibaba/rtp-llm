load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "exec_ops_test",
    srcs = ["ExecOpsTest.cc"],
    deps = [
        "//rtp_llm/cpp/core:exec_ops_test_lib",
        "//rtp_llm/cpp/core:exec_ops_hdr",
        "//rtp_llm/cpp/core:distributed_comm_hdr",
        "//rtp_llm/cpp/cuda/ops:gpu_base",
        "//rtp_llm/cpp/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["manual"],
)

cc_test(
    name = "distributed_comm_test",
    deps = [
        "//rtp_llm/cpp/core:exec_ops_test_lib",
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/cpp/cuda/ops:gpu_base",
        "//rtp_llm/cpp/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["manual"],
)
