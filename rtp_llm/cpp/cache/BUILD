load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "runtime_memory_sizing",
    srcs = ["RuntimeMemorySizing.cc"],
    hdrs = ["RuntimeMemorySizing.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cp_slot_mapper",
    srcs = ["CPSlotMapper.cc"],
    hdrs = ["CPSlotMapper.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_config",
        ":cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "cache_group_type",
    hdrs = [
        "CacheGroupType.h",
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
    name = "kv_cache_specs",
    srcs = [
        "KVCacheSpecDesc.cc",
    ],
    hdrs = [
        "KVCacheSpec.h",
        "KVCacheSpecBase.h",
        "KVCacheSpecDesc.h",
        "MHAKVCacheSpec.h",
        "LinearKVCacheSpec.h",
        "MLAKVCacheSpec.h",
        "OpaqueKVCacheSpec.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_info",
        ":cache_group_type",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:types",
    ],
)

alias(
    name = "kv_cache_spec_desc",
    actual = ":kv_cache_specs",
    visibility = ["//visibility:public"],
)

alias(
    name = "kvcache_spec_desc",
    actual = ":kv_cache_specs",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_config",
    srcs = [
        "CacheConfig.cc",
        "CacheTopology.cc",
    ],
    hdrs = [
        "CacheConfig.h",
        "CacheTopology.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_group_type",
        ":kv_cache_specs",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:types",
    ],
)

cc_library(
    name = "cache_layout",
    hdrs = ["BufferTypes.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_config",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
)

cc_library(
    name = "warm_up_result",
    hdrs = [
        "WarmUpResult.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Host-testable KV sizing implementation. It consumes an explicit MemoryStatus and does not use
# ExecOps or rtp_compute_ops at runtime, so its test can run without a GPU or NVML. cache_config
# and model_config still carry their existing transitive bindings/torch build dependencies. The
# production sampling entry point is provided separately by :memory_evaluation_sizing_device.
cc_library(
    name = "memory_evaluation_sizing",
    srcs = ["MemoryEvaluationHelper.cc"],
    hdrs = ["MemoryEvaluationHelper.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_config",
        ":runtime_memory_sizing",
        ":warm_up_result",
        ":warmup_result_assembly",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:memory_status",
    ],
)

cc_library(
    name = "memory_evaluation_sizing_device",
    srcs = ["MemoryEvaluationHelperDevice.cc"],
    hdrs = ["MemoryEvaluationHelperDevice.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":memory_evaluation_sizing",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ] + torch_deps(),
)

# Pure assembly of the two warmup memory samples into a WarmUpResult, plus the poolShrinkBytes
# helper both the sizing and the logging layer share. Its own target is dependency-free and
# testable without GPU, bindings, or torch.
cc_library(
    name = "warmup_result_assembly",
    srcs = ["WarmUpResultAssembly.cc"],
    hdrs = ["WarmUpResultAssembly.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":warm_up_result",
        "//rtp_llm/cpp/utils:memory_status",
    ],
)

cc_library(
    name = "cache_types",
    srcs = [
        "Types.cc",
    ],
    hdrs = [
        "CacheConfig.h",
        "Types.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_info",
        ":batch_kv_cache_resource",
        ":cache_config",
        ":cache_group_type",
        ":cache_layout",
        ":kv_cache_specs",
        ":warm_up_result",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:types",
    ] + torch_deps(),
)

cc_library(
    name = "recent_cache_key_window",
    srcs = ["RecentCacheKeyWindow.cc"],
    hdrs = ["RecentCacheKeyWindow.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_types",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
    ],
)

cc_library(
    name = "prefill_cache_hit_metrics_reporter",
    srcs = ["PrefillCacheHitMetricsReporter.cc"],
    hdrs = ["PrefillCacheHitMetricsReporter.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_types",
        ":cp_slot_mapper",
        ":recent_cache_key_window",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "block_pool",
    srcs = [
        "BlockCache.cc",
        "BlockPool.cc",
        "MemoryLayoutStrategy.cc",
        "SharedBlockCache.cc",
    ],
    hdrs = [
        "BlockCache.h",
        "BlockPool.h",
        "BlockPoolConfig.h",
        "BlockPoolConfigHelper.h",
        "BlockRefCounter.h",
        "MemoryLayoutStrategy.h",
        "MemoryLayoutConfig.h",
        "SharedBlockCache.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_types",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/models_py/bindings/core:type_convert",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "kv_cache_transfer_planner",
    srcs = [
        "KVCacheTransferPlanner.cc",
    ],
    hdrs = [
        "KVCacheTransferPlanner.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cache_group_type",
    ],
)

cc_library(
    name = "kv_cache_group",
    srcs = [
        "FullKVCacheGroup.cc",
        "KVCacheGroup.cc",
        "LinearKVCacheGroup.cc",
        "SWAKVCacheGroup.cc",
    ],
    hdrs = [
        "FullKVCacheGroup.h",
        "KVCacheGroup.h",
        "LinearKVCacheGroup.h",
        "SWAKVCacheGroup.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_pool",
        ":cache_types",
    ],
)

cc_library(
    name = "kv_cache_allocator_hdr",
    hdrs = [
        "KVCacheAllocator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":block_pool",
        ":cache_types",
        ":cp_slot_mapper",
        "//rtp_llm/cpp/metrics:metrics",
    ] + torch_deps(),
)

cc_library(
    name = "kv_cache_allocator",
    srcs = [
        "HybridKVCacheAllocator.cc",
        "HybridPoolKVCacheAllocator.cc",
        "HybridTypeKVCacheAllocator.cc",
        "KVCacheAllocator.cc",
        "SingleTypeKVCacheAllocator.cc",
    ],
    hdrs = [
        "HybridKVCacheAllocator.h",
        "HybridPoolKVCacheAllocator.h",
        "HybridTypeKVCacheAllocator.h",
        "KVCacheAllocator.h",
        "SingleTypeKVCacheAllocator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cp_slot_mapper",
        ":kv_cache_allocator_hdr",
        ":kv_cache_group",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
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
        ":cache_config",
        ":cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "cache_core",
    srcs = [
        "CacheConfigCreator.cc",
        "HybridConfigCreator.cc",
        "HybridPoolConfigCreator.cc",
        "KVCacheHashUtil.cc",
        "SingleConfigCreator.cc",
    ],
    hdrs = [
        "CacheConfigCreator.h",
        "HybridConfigCreator.h",
        "HybridPoolConfigCreator.h",
        "KVCacheHashUtil.h",
        "SingleConfigCreator.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":batch_kv_cache_resource",
        ":cache_types",
        ":kv_cache_allocator",
        ":memory_evaluation_sizing",
        ":memory_evaluation_sizing_device",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/models_py/bindings/core:type_convert",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        # Device sampling is isolated in :memory_evaluation_sizing_device. cache_core sources
        # include no CUDA/ROCm headers, so this target needs no platform host-utils select.
        # block_pool keeps its cuda_host_utils dep because BlockPool.cc includes cuda_runtime.h.
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
        ":cp_slot_mapper",
        ":prefill_cache_hit_metrics_reporter",
        "//rtp_llm/cpp/cache/connector",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + torch_deps(),
)
