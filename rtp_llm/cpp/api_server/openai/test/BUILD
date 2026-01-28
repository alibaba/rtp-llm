# py_test 和 py_library 已清理，测试通过 pytest 管理

load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/config:config_modules",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_binary(
    name = "openai_unittest_lib.so",
    srcs = [],
    deps = [
        ":openai_unittest_lib",
    ],
    linkshared = 1,
)

cc_library(
    name = "openai_unittest_lib",
    srcs = [
        "APIDataTypeTest.cc",
        "OpenaiEndpointTest.cc",
        "TestMain.cc",
    ] + [
        "//rtp_llm/cpp/api_server/test:api_server_mock_hdrs",
    ],
    deps = test_deps + [
        "//rtp_llm/cpp/api_server:tokenizer",
        "//rtp_llm/cpp/api_server:openai",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    data = [],
    copts = test_copts,
    alwayslink = 1,
)
