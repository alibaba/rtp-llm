load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

py_test (
    name = "model_rpc_client_test",
    srcs = ["model_rpc_client_test.py"],
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    data = [
        "//:th_transformer"
    ],
    exec_properties = {'gpu':'A10'},
)

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    "//rtp_llm/cpp/model_rpc:model_rpc_server",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@havenask//aios/autil:base64",
    "@havenask//aios/autil:zlib",
] + torch_deps() + device_impl_target()

# cc_test(
#     name = "model_rpc_server_test",
#     srcs = [
#         "RpcServerTest.cc",
#     ],
#     copts = test_copts,
#     deps = [
#         "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
#         "//rtp_llm/cpp:components",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "//rtp_llm/cpp/model_rpc:model_rpc_server",
#         "//rtp_llm/cpp/utils:utils",
#     ] + torch_deps(),
# )

cc_test(
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "broadcast_manager_test",
    srcs = [
        "BroadcastManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "@havenask//aios/autil:net",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)
