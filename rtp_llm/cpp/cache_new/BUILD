load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "batch_kv_cache_resource",
    hdrs = [
        "BatchKVCacheResource.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

cc_library(
    name = "cache_new_types",
    hdrs = [
        "CacheConfig.h",
        "types.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:types",
    ] + torch_deps(),
)

cc_library(
    name = "block_pool",
    srcs = [
        "BlockCache.cc",
        "BlockPool.cc",
        "MemoryLayoutStrategy.cc",
    ],
    hdrs = [
        "BlockCache.h",
        "BlockPool.h",
        "BlockPoolConfigHelper.h",
        "BlockRefCounter.h",
        "MemoryLayoutStrategy.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_new_types",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:lru_cache",
    ],
)

cc_library(
    name = "kv_cache_group",
    srcs = [
        "FullKVCacheGroup.cc",
        "KVCacheGroup.cc",
        "LinearKVCacheGroup.cc",
    ],
    hdrs = [
        "FullKVCacheGroup.h",
        "KVCacheGroup.h",
        "LinearKVCacheGroup.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_pool",
    ],
)

cc_library(
    name = "kv_cache_allocator",
    srcs = [
        "HybridLayerKVCacheAllocator.cc",
        "KVCacheAllocator.cc",
        "SingleTypeKVCacheAllocator.cc",
    ],
    hdrs = [
        "HybridLayerKVCacheAllocator.h",
        "KVCacheAllocator.h",
        "SingleTypeKVCacheAllocator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_group",
    ],
)

cc_library(
    name = "connector_base",
    hdrs = [
        "AsyncContext.h",
        "KVCacheConnector.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "batch_kv_cache_resource",
    ],
)

cc_library(
    name = "memory_connector",
    srcs = [
        "KVCacheMemoryConnector.cc",
        "MemoryBlockCache.cc",
    ],
    hdrs = [
        "KVCacheMemoryConnector.h",
        "MemoryBlockCache.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":connector_base",
        "//rtp_llm/cpp/cache_new:kv_cache_allocator",
        "//rtp_llm/cpp/model_rpc:tp_broadcast_manager",
    ],
)

cc_library(
    name = "connector_coordinator",
    srcs = [
        "KVCacheConnectorCoordinator.cc",
    ],
    hdrs = [
        "KVCacheConnectorCoordinator.h",
        "KVCacheConnectorReadWriteContext.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":memory_connector",
        "//rtp_llm/cpp/cache_new/remote_connector",
    ],
)

cc_library(
    name = "cache_new",
    srcs = [
        "CacheConfigCreator.cc",
        "KVCacheHashUtil.cc",
        "KVCacheManager.cc",
    ],
    hdrs = [
        "CacheConfigCreator.h",
        "KVCacheHashUtil.h",
        "KVCacheManager.h",
        "WarmUpResult.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_new_types",
        ":connector_coordinator",
        ":kv_cache_allocator",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps() + select({
        "//:using_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "storage_3fs",
    srcs = [
    ],
    hdrs = [
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//3rdparty/3fs:hf3fs",
    ],
)
