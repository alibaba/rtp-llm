load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp:utils",
    "//rtp_llm/cpp:normal_engine",
    "//rtp_llm/cpp:schedulers",
    "//:gpt_init_params",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "fifo_scheduler_test",
    srcs = [
        "FIFOSchedulerTest.cc",
    ],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
