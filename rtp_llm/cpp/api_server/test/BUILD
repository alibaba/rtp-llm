# py_test 和 py_library 已清理，测试通过 pytest 管理

load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + device_impl_target()

cc_binary(
    name = "api_server_unittest_lib.so",
    srcs = [],
    deps = [
        ":api_server_unittest_lib",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    linkshared = 1,
)

cc_library(
    name = "api_server_unittest_lib",
    srcs = [ # TODO: use glob
        "mock/MockEngineBase.h",
        "mock/MockHttpResponseWriter.h",
        "mock/MockTokenProcessor.h",
        "mock/MockApiServerMetricReporter.h",
        "mock/MockGenerateStream.h",
        "mock/MockGenerateStreamWrapper.h",
        "mock/MockEmbeddingEndpoint.h",
        "mock/MockChatRender.h",
        "mock/MockTokenizer.h",
        "mock/MockOpenaiEndpoint.h",
        "mock/MockLoraManager.h",
        "mock/MockWeightsLoader.h",
        "ErrorResponseTest.cc",
        "HealthServiceTest.cc",
        "HttpApiServerTest.cc",
        "WorkerStatusServiceTest.cc",
        "ModelStatusServiceTest.cc",
        "ParallelInfoTest.cc",
        "SysCmdServiceTest.cc",
        "TokenizerEncodeResponseTest.cc",
        "TokenizerServiceTest.cc",
        "GenerateStreamWrapperTest.cc",
        "InferenceServiceTest.cc",
        "EmbeddingServiceTest.cc",
        "InferenceDataTypeTest.cc",
        "ConcurrencyControllerTest.cc",
        "ChatServiceTest.cc",
        "TestMain.cc",
    ],
    deps = test_deps + [
        "//rtp_llm/cpp/api_server:http_api_server",
    ],
    copts = test_copts,
    alwayslink = 1,
)

filegroup(
    name = "api_server_mock_hdrs",
    srcs = [
        "mock/MockChatRender.h",
        "mock/MockTokenizer.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "testdata",
    srcs = [
        "testdata/annocation",
        "testdata/parallel.json",
    ],
    visibility = ["//visibility:public"],
)
