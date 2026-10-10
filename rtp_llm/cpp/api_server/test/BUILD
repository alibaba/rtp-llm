load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

test_deps = [
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
        "@//:using_rocm": {"TEST_USING_DEVICE": "ROCM"},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA"},
    }),
    exec_properties = {'gpu':'H20'},
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
    data = ["//:rtp_compute_ops"],
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

cc_test(
    name = "constraint_tree_service_test",
    srcs = ["ConstraintTreeServiceTest.cc", "mock/MockHttpResponseWriter.h"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/api_server:constraint_tree_service",
        "//rtp_llm/cpp/api_server:http_client",
        "//rtp_llm/cpp/api_server/common:common_service",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps + device_impl_target() + select({
        "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
        "//conditions:default": ["//rtp_llm/models_py/bindings/core:exec_ops_test_lib"],
    }),
)
cc_binary(
    name = "constraint_tree_test_server",
    srcs = ["ConstraintTreeTestServer.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/api_server:constraint_tree_service",
        "//rtp_llm/cpp/api_server/common:common_service",
    ] + cuda13_torch_link_deps,
)
cc_binary(
    name = "tokenizer_sid_mapping_test_lib.so",
    srcs = ["TokenizerSidMappingTestBinding.cc"],
    linkshared = 1,
    deps = ["//rtp_llm/cpp/api_server:tokenizer"],
)
py_test(
    name = "tokenizer_sid_mapping_test",
    srcs = ["tokenizer_sid_mapping_test.py"],
    imports = ["."],
    data = [":tokenizer_sid_mapping_test_lib.so", "//rtp_llm:frontend"],
    deps = ["//rtp_llm:testlib"],
)
