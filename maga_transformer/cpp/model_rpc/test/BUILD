load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//maga_transformer/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + copts()

# py_test (
#     name = "model_rpc_client_test",
#     srcs = ["model_rpc_client_test.py"],
#     deps = [
#         "//maga_transformer/cpp:model_rpc_client",
#         "//maga_transformer:testlib",
#         "//maga_transformer/test/model_test/test_util:test_util"
#     ],
# )

test_deps = [
    "//maga_transformer/cpp/devices/testing:device_test_utils",
    "//maga_transformer/cpp:models",
    "//:gpt_init_params",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + device_impl_target()

# cc_test(
#     name = "model_rpc_server_test",
#     srcs = [
#         "RpcServerTest.cc",
#     ],
#     copts = test_copts,
#     deps = [
#         "//maga_transformer/cpp/proto:model_rpc_service_cc_proto",
#         "//maga_transformer/cpp:components",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "//maga_transformer/cpp:model_rpc_server",
#         "//maga_transformer/cpp:utils",
#     ] + torch_deps(),
# )

cc_test(
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//maga_transformer/cpp/proto:model_rpc_service_cc_proto",
        "//maga_transformer/cpp:model_rpc_server",
        "//:gpt_init_params",
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp/devices/cuda_impl:cuda_impl",
        "//maga_transformer/cpp/devices/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main"
    ] + torch_deps(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

