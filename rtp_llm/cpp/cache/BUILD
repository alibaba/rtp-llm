load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache",
    hdrs = [
        "BatchKVCacheResource.h",
        "BlockCache.h",
        "BlockRefCounter.h",
        "CacheConfigCreator.h",
        "CacheConfig.h",
        "CacheManager.h",
        "KVCacheResource.h",
        "KvCacheInfo.h",
        "DistStorage.h",
        "DistStorageLocalMem.h",
        "DistStorageManager.h",
        "DistKvCachePlanner.h",
        "DistKvCache.h",
        "DistKvCacheMetrics.h",
        "KVCacheAllocator.h",
        "BlockLRUCache.h",
        "WarmUpResult.h",
        "MemoryBlockCache.h",
    ],
    srcs = [
        "BatchKVCacheResource.cc",
        "BlockCache.cc",
        "CacheConfigCreator.cc",
        "CacheManager.cc",
        "KVCacheResource.cc",
        "DistStorageLocalMem.cc",
        "DistStorageManager.cc",
        "DistKvCachePlanner.cc",
        "DistKvCache.cc",
        "KVCacheAllocator.cc",
        "BlockLRUCache.cc",
        "MemoryBlockCache.cc",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/metrics:metrics",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/cache_new:cache_new_types",
    ] + select({
        "//:using_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
    copts = copts() + select({
        "//:using_3fs": ["-DENABLE_3FS=1"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "storage_3fs",
    hdrs = [
        "DistStorage.h",
        "DistStorage3FS.h",
        "DistStorage3FSFile.h",
        "ThreeFSCudaUtil.h",
        "ThreeFSMempool.h",
        "DistKvCacheMetrics.h",
    ],
    srcs = [
        "DistStorage3FS.cc",
        "DistStorage3FSFile.cc",
        "ThreeFSMempool.cc",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:lru_cache",
        "//3rdparty/3fs:hf3fs",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
    ],
    visibility = ["//visibility:public"],
)