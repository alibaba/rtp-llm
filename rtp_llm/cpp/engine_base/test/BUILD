load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "cpu_quiesce_coordinator_test",
    srcs = ["CpuQuiesceCoordinatorTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:cpu_quiesce_coordinator",
        "//rtp_llm/cpp/engine_base/sleep",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_python//:python_lib",
    ] + torch_deps() + cuda13_torch_link_deps,
    # Test-only: several Gloo ranks share one interpreter, unlike production.
    # Torch 2.11's FlightRecorder does not serialize this concurrent bootstrap.
    env = {"TORCH_FR_BUFFER_SIZE": "0"},
    timeout = "short",
)

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
