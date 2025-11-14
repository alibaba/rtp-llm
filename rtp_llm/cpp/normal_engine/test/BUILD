load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()


cc_library(
    name = "mock_engine",
    hdrs = [
        "MockEngine.h"
    ],
    srcs = [
    ],
    copts = test_copts,
    deps =  [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/engine_base:weights_converter",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "batch_stream_processor_test",
    srcs = glob([
        "NormalBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "engine_test",
    srcs = glob([
        "EngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps =  [
        ":mock_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
