load("//:def.bzl", "copts")

cc_library(
    name = "connector_base",
    hdrs = [
        "KVCacheConnector.h",
        "KVCacheConnectorLayerContext.h",
        "Meta.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/utils:error_code",
    ],
)

cc_library(
    name = "connector_context",
    hdrs = [
        "KVCacheConnectorReadWriteContext.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":connector_base",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
    ],
)
