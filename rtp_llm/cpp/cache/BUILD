load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache",
    srcs = [
        "BatchKVCacheResource.cc",
        "BlockCache.cc",
        "BlockLRUCache.cc",
        "CacheConfigCreator.cc",
        "CacheManager.cc",
        "DistKvCache.cc",
        "DistKvCachePlanner.cc",
        "DistStorageLocalMem.cc",
        "DistStorageManager.cc",
        "KVCacheAllocator.cc",
        "KVCacheResource.cc",
        "MemoryBlockCache.cc",
    ],
    hdrs = [
        "BatchKVCacheResource.h",
        "BlockCache.h",
        "BlockLRUCache.h",
        "BlockRefCounter.h",
        "CacheConfig.h",
        "CacheConfigCreator.h",
        "CacheManager.h",
        "DistKvCache.h",
        "DistKvCacheMetrics.h",
        "DistKvCachePlanner.h",
        "DistStorage.h",
        "DistStorageLocalMem.h",
        "DistStorageManager.h",
        "KVCacheAllocator.h",
        "KVCacheResource.h",
        "KvCacheInfo.h",
        "MemoryBlockCache.h",
        "WarmUpResult.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/cache2/kv_cache_connector",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:lru_cache",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "//:enable_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "storage_3fs",
    srcs = [
        "DistStorage3FS.cc",
        "DistStorage3FSFile.cc",
        "ThreeFSMempool.cc",
    ],
    hdrs = [
        "DistKvCacheMetrics.h",
        "DistStorage.h",
        "DistStorage3FS.h",
        "DistStorage3FSFile.h",
        "ThreeFSCudaUtil.h",
        "ThreeFSMempool.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//3rdparty/3fs:hf3fs",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/utils:lru_cache",
    ],
)
