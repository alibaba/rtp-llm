load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

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

py_test(
    name = "openai_unittest",  # 需要有一个同名 py 文件
    srcs = [
        "openai_unittest.py",
    ],
    data = [],
    env = select({
        "//:using_rocm": {"TEST_USING_DEVICE": "ROCM"},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA"},
    }),
    exec_properties = {"gpu": "A10_CU13"},
    imports = [
        ".",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["A10_CU13"],
    deps = [
        ":openai_unittest_pylib",
    ],
)

py_library(
    name = "openai_unittest_pylib",
    srcs = [],
    data = [
        ":openai_unittest_lib.so",
    ],
    deps = [],
)

cc_binary(
    name = "openai_unittest_lib.so",
    srcs = [],
    linkshared = 1,
    deps = [
        ":openai_unittest_lib",
    ],
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
    copts = test_copts,
    data = [],
    deps = test_deps + [
        "//rtp_llm/cpp/api_server:openai",
        "//rtp_llm/cpp/api_server:tokenizer",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    alwayslink = 1,
)
