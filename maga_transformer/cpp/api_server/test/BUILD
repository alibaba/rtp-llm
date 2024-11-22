load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//src/fastertransformer/devices:device_defs.bzl", "device_impl_target", "device_test_envs", "device_linkopts")
 
test_copts = [
    "-fno-access-control",
] + copts()
 
test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
] + device_linkopts()
 
cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)
 
test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//maga_transformer/cpp:models",
    "//:gpt_init_params",
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
    # data = [
    #     "testdata/annocation",
    #     "testdata/parallel.json",
    # ],
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
    ],
    linkshared = 1,
)

cc_library(
    name = "api_server_unittest_lib",
    srcs = [
        "mock/MockEngineBase.h",
        "mock/MockHttpResponseWriter.h",
        "mock/MockTokenProcessor.h",
        "ErrorResponseTest.cc",
        "HealthServiceTest.cc",
        "HttpApiServerTest.cc",
        "WorkerStatusServiceTest.cc",
        "ModelStatusServiceTest.cc",
        "ParallelInfoTest.cc",
        "SysCmdServiceTest.cc",
        "TokenizerEncodeResponseTest.cc",
        "TokenizerServiceTest.cc",
        "TestBinding.cc",
    ],
    deps = test_deps + [
        "//maga_transformer/cpp:http_api_server",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    alwayslink = 1,
)
