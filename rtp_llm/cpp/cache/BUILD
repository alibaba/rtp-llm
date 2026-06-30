load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache_group_type",
    hdrs = [
        "spec/CacheGroupType.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "block_info",
    hdrs = [
        "BlockInfo.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_types",
    hdrs = [
        "BufferTypes.h",
        "CacheConfig.h",
        "spec/CacheGroupType.h",
        "spec/KVCacheSpec.h",
        "spec/KVCacheSpecBase.h",
        "spec/MHAKVCacheSpec.h",
        "spec/LinearKVCacheSpec.h",
        "spec/MLAKVCacheSpec.h",
        "Types.h",
        "WarmUpResult.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_info",
        ":batch_kv_cache_resource",
        ":cache_group_type",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings/core:types",
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
        "MemoryLayoutConfig.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/utils:profiling_scope",
    ],
)

cc_library(
    name = "kv_cache_group",
    srcs = [
        "group/FullKVCacheGroup.cc",
        "group/KVCacheGroup.cc",
        "group/LinearKVCacheGroup.cc",
    ],
    hdrs = [
        "group/FullKVCacheGroup.h",
        "group/KVCacheGroup.h",
        "group/LinearKVCacheGroup.h",
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
        "allocator/HybridTypeKVCacheAllocator.cc",
        "allocator/KVCacheAllocator.cc",
        "allocator/SingleTypeKVCacheAllocator.cc",
    ],
    hdrs = [
        "allocator/HybridTypeKVCacheAllocator.h",
        "allocator/KVCacheAllocator.h",
        "allocator/SingleTypeKVCacheAllocator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_group",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
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
        "//rtp_llm/cpp/cache:cache_group_type",
    ],
)

cc_library(
    name = "cache_core",
    srcs = [
        "config_creator/CacheConfigCreator.cc",
        "config_creator/HybridConfigCreator.cc",
        "KVCacheHashUtil.cc",
        "config_creator/MemoryEvaluationHelper.cc",
        "config_creator/SingleConfigCreator.cc",
    ],
    hdrs = [
        "config_creator/CacheConfigCreator.h",
        "config_creator/HybridConfigCreator.h",
        "KVCacheHashUtil.h",
        "config_creator/MemoryEvaluationHelper.h",
        "config_creator/SingleConfigCreator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_types",
        ":kv_cache_allocator",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:type_convert",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
        ],
        "//conditions:default": [],
    }) + select({
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
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + torch_deps(),
)
