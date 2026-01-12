load("//:def.bzl", "copts")

load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

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

cc_test(
    name = "components_test",
    srcs = [
        "main.cc",
        "TPBroadcastClientTest.cc",
        "P2PConnectorServerCallerTest.cc",
        "AsymmetricTpUtilTest.cc",
        "PrefillWorkerLoadContextTest.cc",
        "ComputedLayerCacheBufferTest.cc",
        "StoreWaitContextTest.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:components",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_scheduler_test",
    srcs = [
        "P2PConnectorSchedulerTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:connector",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_worker_test",
    srcs = [
        "P2PConnectorWorkerTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:connector",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)
