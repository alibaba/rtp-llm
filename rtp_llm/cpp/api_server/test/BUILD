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

# 通过 python 执行 cpp 单测
py_test(
    name = "api_server_unittest",  # 需要有一个同名 py 文件
    python_version = 'PY3',
    srcs_version = 'PY3',
    srcs = [
        "api_server_unittest.py",
    ],
    deps = [
        ":api_server_unittest_pylib",
    ],
    data = [
        "testdata/annocation",
        "testdata/parallel.json",
    ],
    imports = [
        ".",
    ],
    env = select({
        "@//:using_rocm": {"TEST_USING_DEVICE": "ROCM", "DEVICE_RESERVE_MEMORY_BYTES": "128"},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA", "DEVICE_RESERVE_MEMORY_BYTES": "128"},
    }),
    exec_properties = {'gpu':'A10'},
)

py_library(
    name = "api_server_unittest_pylib",
    srcs = [],
    deps = [],
    data = [
        ":api_server_unittest_lib.so",
    ],
)

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
