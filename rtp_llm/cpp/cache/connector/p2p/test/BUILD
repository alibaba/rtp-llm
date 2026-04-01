load("//:def.bzl", "copts")

load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_rpc_server",
    srcs = [
        "TestRpcServer.cc",
    ],
    hdrs = [
        "TestRpcServer.h",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
    copts = test_copts,
)

cc_library(
    name = "mock_generate_stream",
    hdrs = [
        "MockGenerateStream.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector:connector",
    ],
    copts = test_copts,
)

cc_test(
    name = "components_test",
    srcs = [
        "main.cc",
        "P2PBroadcastClientTest.cc",
        "PrefillLoadCallerTest.cc",
        "AsymmetricTpUtilTest.cc",
        "ComputedLayerCacheBufferTest.cc",
        "StoreWaitContextTest.cc",
        "P2PConnectorResourceStoreTest.cc",
    ],
    deps = [
        ":test_rpc_server",
        ":mock_generate_stream",
        "//rtp_llm/cpp/cache/connector/p2p:components",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "p2p_connector_test",
    srcs = [
        "P2PConnectorTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        ":mock_generate_stream",
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "p2p_connector_scheduler_test",
    srcs = [
        "P2PConnectorSchedulerTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        ":mock_generate_stream",
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "p2p_connector_worker_test",
    srcs = [
        "P2PConnectorWorkerTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)
