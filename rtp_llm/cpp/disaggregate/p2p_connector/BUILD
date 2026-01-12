package(default_visibility = ["//rtp_llm/cpp/disaggregate/p2p_connector:__subpackages__"])

cc_library(
    name = "layer_block_convertor",
    srcs = [
    ],
    hdrs = [
        "LayerBlockConvertorImpl.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
    ],
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
        "TPBroadcastClient.cc",
        "P2PConnectorServerCaller.cc",
        "AsymmetricTpUtil.cc",
        "PrefillWorkerLoadContext.cc",
        "ComputedLayerCacheBuffer.cc",
        "P2PConnectorStreamStore.cc",
        "StoreWaitContext.cc",
    ],
    hdrs = [
        "TPBroadcastClient.h",
        "P2PConnectorServerCaller.h",
        "AsymmetricTpUtil.h",
        "PrefillWorkerLoadContext.h",
        "ComputedLayerCacheBuffer.h",
        "P2PConnectorStreamStore.h",
        "StoreWaitContext.h",
    ],
    deps = [
        ":metrics",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/core:event",
        "//rtp_llm/cpp/model_rpc:tp_broadcast_manager",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/disaggregate/transfer:basic_types",
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
        "P2PConnectorWorker.cc",
        "P2PConnector.cc",
    ],
    hdrs = [
        "P2PConnectorAsyncContext.h",
        "P2PConnectorScheduler.h",
        "P2PConnectorWorker.h",
        "P2PConnector.h",
    ],
    deps = [
        ":components",
        ":metrics",
        "//rtp_llm/cpp/cache/connector:connector_header",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/autil:thread",
        "//rtp_llm/cpp/disaggregate/transfer:transfer_server",
        "//rtp_llm/cpp/disaggregate/transfer:transfer_client",
    ],
    visibility = ["//visibility:public"],
)
