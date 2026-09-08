load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//rtp_llm/cpp/cache/connector/p2p:__subpackages__"])

cc_library(
    name = "p2p_connector_config",
    hdrs = ["P2PConnectorConfig.h"],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_config",
        "//rtp_llm/cpp/config:config_modules",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "layer_cache_buffer",
    hdrs = [
        "LayerCacheBuffer.h",
        "LayerCacheBufferUtil.h",
    ],
    srcs = [
        "LayerCacheBuffer.cc",
        "LayerCacheBufferUtil.cc",
    ],
    deps = [
        ":layer_block_converter",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "@havenask//aios/autil:thread",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "layer_block_converter",
    srcs = [
    ],
    hdrs = [
        "LayerBlockConverter.h",
        "LayerBlockConverterImpl.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "metrics",
    srcs = [
        "P2PConnectorMetrics.cc",
    ],
    hdrs = [
        "P2PConnectorMetrics.h",
    ],
    deps = [
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:time_util",
    ]
)

cc_library(
    name = "components",
    srcs = [
        "P2PBroadcastClient.cc",
        "PrefillLoadCaller.cc",
        "AsymmetricTpUtil.cc",
        "ComputedLayerCacheBuffer.cc",
        "P2PConnectorResourceStore.cc",
        "StoreWaitContext.cc",
    ],
    hdrs = [
        "P2PBroadcastClient.h",
        "PrefillLoadCaller.h",
        "AsymmetricTpUtil.h",
        "ComputedLayerCacheBuffer.h",
        "P2PConnectorResourceStore.h",
        "StoreWaitContext.h",
    ],
    deps = [
        ":layer_cache_buffer",
        ":metrics",
        "//rtp_llm/cpp/cache/connector:connector_header",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/autil:thread",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "connector",
    srcs = [
        "P2PConnectorAsyncContext.cc",
        "P2PConnectorScheduler.cc",
        "P2PConnectorSchedulerPrefill.cc",
        "P2PConnectorSchedulerDecode.cc",
        "P2PConnectorWorker.cc",
        "P2PConnectorWorkerPrefill.cc",
        "P2PConnectorWorkerDecode.cc",
        "P2PConnector.cc",
    ],
    hdrs = [
        "P2PConnectorAsyncContext.h",
        "P2PConnectorConfig.h",
        "P2PConnectorScheduler.h",
        "P2PConnectorSchedulerPrefill.h",
        "P2PConnectorSchedulerDecode.h",
        "P2PConnectorWorker.h",
        "P2PConnectorWorkerPrefill.h",
        "P2PConnectorWorkerDecode.h",
        "P2PConnector.h",
        "P2PKeyUtil.h",
    ],
    deps = [
        ":components",
        ":layer_block_converter",
        ":layer_cache_buffer",
        ":metrics",
        ":p2p_connector_config",
        "//rtp_llm/cpp/cache/connector:connector_header",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/autil:thread",
        "@havenask//aios/autil:net",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:backend_factory",
    ],
    visibility = ["//visibility:public"],
)
