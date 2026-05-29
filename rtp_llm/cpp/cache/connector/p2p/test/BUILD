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

py_test(
    name = "pd_kv_writeback_p2p_static_test",
    srcs = ["PdKvWritebackP2PStaticTest.py"],
    data = [
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorker.cc",
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorker.h",
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorkerDecode.cc",
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorkerDecode.h",
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorkerPrefill.cc",
        "//rtp_llm/cpp/cache/connector/p2p:P2PConnectorWorkerPrefill.h",
        "//rtp_llm/cpp/cache/connector/p2p:LayerBlockConverterImpl.h",
        "//rtp_llm/cpp/cache:BlockPool.h",
        "//rtp_llm/cpp/cache/writeback:PdKvWritebackTransfer.h",
    ],
    main = "PdKvWritebackP2PStaticTest.py",
)

cc_library(
    name = "mock_generate_stream",
    hdrs = [
        "MockGenerateStream.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector:connector",
        "//rtp_llm/cpp/engine_base/stream:stream",
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
        "//rtp_llm/cpp/engine_base/stream:stream",
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
