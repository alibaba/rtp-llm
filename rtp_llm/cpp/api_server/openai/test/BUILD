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

py_test(
    name = "openai_unittest",  # 需要有一个同名 py 文件
    python_version = 'PY3',
    srcs_version = 'PY3',
    srcs = [
        "openai_unittest.py",
    ],
    deps = [
        ":openai_unittest_pylib",
    ],
    data = [],
    imports = [
        ".",
    ],
    env = select({
        "//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
    exec_properties = {'gpu':'A10'},
)

py_library(
    name = "openai_unittest_pylib",
    srcs = [],
    deps = [],
    data = [
        ":openai_unittest_lib.so",
    ],
)

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