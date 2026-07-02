load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_copts = copts()

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
    exec_properties = {'gpu':'H20'},
)

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
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
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "rpc_timeout_utils_test",
    srcs = [
        "RpcTimeoutUtilsTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "decode_rpc_server_new2_test",
    srcs = [
        "DecodeRpcServerNew2Test.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "prefill_rpc_server_new2_test",
    srcs = [
        "PrefillRpcServerNew2Test.cc",
    ],
    copts = test_copts + ["-fno-access-control"],
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "broadcast_manager_test",
    srcs = [
        "BroadcastManagerTest.cc",
    ],
    data = [],
    copts = test_copts + ["-fno-access-control"],
    deps = test_deps + [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "@havenask//aios/autil:net",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "prefill_server_caller_context_test",
    srcs = [
        "PrefillServerCallerTest.cc",
    ],
    copts = test_copts + ["-fno-access-control"],
    deps = [
        "//rtp_llm/cpp/model_rpc:prefill_server_caller",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:grpc_address_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "prefill_rpc_server_test",
    srcs = [
        "PrefillRpcServerTest.cc",
    ],
    copts = test_copts + ["-fno-access-control"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/normal_engine:normal_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)
