load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache_group_type",
    hdrs = [
        "CacheGroupType.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_types",
    hdrs = [
        "BufferTypes.h",
        "CacheConfig.h",
        "CacheGroupType.h",
        "KVCacheSpec.h",
        "MemoryLayout.h",
        "Types.h",
        "WarmUpResult.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_group_type",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
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
        "BlockPoolConfig.h",
        "BlockPoolConfigHelper.h",
        "BlockRefCounter.h",
        "MemoryLayoutStrategy.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_types",
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
        "HybridKVCacheAllocator.cc",
        "KVCacheAllocator.cc",
        "SingleTypeKVCacheAllocator.cc",
    ],
    hdrs = [
        "HybridKVCacheAllocator.h",
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
    name = "batch_kv_cache_resource",
    srcs = ["KVCacheResource.cc"],
    hdrs = [
        "BatchKVCacheResource.h",
        "KVCacheResource.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "cache_core",
    srcs = [
        "CacheConfigCreator.cc",
        "HybridConfigCreator.cc",
        "KVCacheHashUtil.cc",
        "MemoryEvaluationHelper.cc",
        "SingleConfigCreator.cc",
    ],
    hdrs = [
        "CacheConfigCreator.h",
        "HybridConfigCreator.h",
        "KVCacheHashUtil.h",
        "MemoryEvaluationHelper.h",
        "SingleConfigCreator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_types",
        ":kv_cache_allocator",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps() + select({
        "//:using_remote_kv_cache": [
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

cc_library(
    name = "cache",
    srcs = [
        "KVCacheManager.cc",
    ],
    hdrs = [
        "KVCacheManager.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_core",
        "//rtp_llm/cpp/cache/connector",
    ] + torch_deps(),
)
