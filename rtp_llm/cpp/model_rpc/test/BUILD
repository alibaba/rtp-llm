load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + copts()

# py_test (
#     name = "model_rpc_client_test",
#     srcs = ["model_rpc_client_test.py"],
#     deps = [
#         "//rtp_llm/cpp:model_rpc_client",
#         "//rtp_llm:testlib",
#         "//rtp_llm/test/model_test/test_util:test_util"
#     ],
# )

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp:models",
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
#         "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
#         "//rtp_llm/cpp:components",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "//rtp_llm/cpp:model_rpc_server",
#         "//rtp_llm/cpp:utils",
#     ] + torch_deps(),
# )

cc_test(
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp:model_rpc_server",
        "//:gpt_init_params",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main"
    ] + torch_deps(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

