load("//:def.bzl", "copts")
load("//bazel:device_defs.bzl", "device_impl_target")

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
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
)

cc_library(
    name = "mock_generate_stream",
    hdrs = [
        "MockGenerateStream.h",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p:legacy_support",
        "//rtp_llm/cpp/engine_base/stream",
    ],
)

cc_test(
    name = "components_test",
    srcs = [
        "AsymmetricTpUtilTest.cc",
        "ComputedLayerCacheBufferTest.cc",
        "P2PBroadcastClientTest.cc",
        "P2PConnectorResourceStoreTest.cc",
        "PrefillLoadCallerTest.cc",
        "StoreWaitContextTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":mock_generate_stream",
        ":test_rpc_server",
        "//rtp_llm/cpp/cache/connector/p2p:components",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

cc_test(
    name = "p2p_connector_test",
    srcs = [
        "P2PConnectorTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":mock_generate_stream",
        ":test_rpc_server",
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/engine_base/stream",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

cc_test(
    name = "p2p_connector_scheduler_test",
    srcs = [
        "P2PConnectorSchedulerTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":mock_generate_stream",
        ":test_rpc_server",
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

cc_test(
    name = "p2p_connector_worker_test",
    srcs = [
        "P2PConnectorWorkerTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)
