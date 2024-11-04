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
 
cc_test(
    name = "api_server_unittest",
    srcs = [
        "mock/MockHttpResponseWriter.h",
        "HealthServiceTest.cc",
        "HttpApiServerTest.cc",
    ],
    deps = test_deps + [
        "//maga_transformer/cpp:http_api_server",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    env = select({
        "//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
    exec_properties = {'gpu':'A10'},
)
