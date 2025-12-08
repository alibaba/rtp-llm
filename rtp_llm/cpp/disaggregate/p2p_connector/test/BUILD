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
    name = "tp_broadcast_test",
    srcs = [
        "TPBroadcastClientTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:tp_broadcast",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "prefill_load_test",
    srcs = [
        "PrefillLoadClientTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:prefill_load_client",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_decode_worker_test",
    srcs = [
        "P2PConnectorDecodeWorkerTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:decode_connector",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_decode_scheduler_test",
    srcs = [
        "P2PConnectorDecodeSchedulerTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:decode_connector",
        "//rtp_llm/cpp/cache_new:cache_new",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_prefill_worker_test",
    srcs = [
        "P2PConnectorPrefillWorkerTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:prefill_connector",
        "//rtp_llm/cpp/cache_new:cache_new",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "p2p_connector_prefill_scheduler_test",
    srcs = [
        "P2PConnectorPrefillSchedulerTest.cc",
        "main.cc",
    ],
    deps = [
        ":test_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:prefill_connector",
        "//rtp_llm/cpp/cache_new:cache_new",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "asymmetric_tp_util_test",
    srcs = [
        "AsymmetricTpUtilTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:asymmetric_tp",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "computed_layer_cache_buffer_store_test",
    srcs = [
        "ComputedLayerCacheBufferStoreTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:computed_layer_cache_buffer",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ]+ device_impl_target(),
    copts = test_copts,
)

cc_test(
    name = "prefill_worker_load_context_test",
    srcs = [
        "PrefillWorkerLoadContextTest.cc",
        "main.cc",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/p2p_connector:prefill_worker_load_context",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ] + device_impl_target(),
    copts = test_copts,
)