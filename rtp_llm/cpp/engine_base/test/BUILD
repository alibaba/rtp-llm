load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

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
