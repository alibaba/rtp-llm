package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mooncake_transfer_service",
    hdrs = [
        "MooncakeRemoteDescriptor.h",
        "MooncakeTransferService.h",
    ],
    srcs = ["MooncakeTransferService.cc"],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto:mooncake_service_cc_proto",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "mooncake_transfer_engine_adapter_provider",
    hdrs = ["MooncakeTransferEngineAdapter.h", "MooncakeTransferEngineAdapterProvider.h"],
    srcs = select({
        "//:enable_mooncake_te": ["MooncakeTransferEngineAdapterClassic.cc"],
        "//conditions:default": ["MooncakeTransferEngineAdapterStub.cc"],
    }),
    copts = select({
        "//:enable_mooncake_te": [
            "-I/data0/qiongshi.gb/mooncake/Mooncake/mooncake-common/include",
            "-I/data0/qiongshi.gb/mooncake/Mooncake/mooncake-transfer-engine/include",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//:enable_mooncake_te": [
            "-L/data0/qiongshi.gb/mooncake/Mooncake/build-rtp/install/lib",
            "-Wl,-rpath,/data0/qiongshi.gb/mooncake/Mooncake/build-rtp/install/lib",
            "-lasio",
            "-lmooncake_common",
            "-ltransfer_engine",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_config",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "mooncake_backend_stub",
    hdrs = [
        "MooncakeKVCacheReceiver.h",
        "MooncakeKVCacheSender.h",
        "MooncakeTransferEngineAdapter.h",
        "MooncakeTransferEngineAdapterProvider.h",
    ],
    srcs = [
        "MooncakeKVCacheReceiver.cc",
        "MooncakeKVCacheSender.cc",
    ],
    deps = [
        ":mooncake_transfer_engine_adapter_provider",
        ":mooncake_transfer_service",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_config",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto:mooncake_service_cc_proto",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/network/arpc:arpc",
    ],
)

cc_test(
    name = "mooncake_backend_stub_test",
    srcs = ["MooncakeBackendStubTest.cc"],
    deps = [
        ":mooncake_backend_stub",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
