load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps", "torch_nvshmem_deps")

cc_test = cc_test_wrapper

cuda13_torch_link_deps = torch_nvshmem_deps()

cc_test(
    name = "step_window_profiler_configure_test",
    srcs = [
        "StepWindowProfilerConfigureTest.cc",
    ],
    copts = [
        "-fno-access-control",
    ] + copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test(
    name = "torch_profiler_test",
    srcs = ["TorchProfilerTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:profiler",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "FT_SERVER_TEST": "1",
    },
)
