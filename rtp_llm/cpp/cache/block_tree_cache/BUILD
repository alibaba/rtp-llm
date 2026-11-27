load("//:def.bzl", "copts")
load(":native_sources.bzl", "BTC_NATIVE_SOURCES", "BTC_NATIVE_HEADERS")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_library(
    name = "tier_watermark",
    hdrs = ["evict/TierWatermark.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "malloc_status",
    hdrs = ["block_pool/MallocStatus.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "block_tree_key",
    hdrs = ["BlockTreeKey.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "prefix_coverage",
    hdrs = ["group_set/PrefixCoverage.h"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/cpp/cache:cache_group_type"],
)

# The standalone shared library and the engine use one implementation of
# BlockTreeCache. Reuse the existing small native targets where they already
# own a translation unit, so a worker linking those targets gets one definition.
cc_library(
    name = "btc_core",
    srcs = [src for src in BTC_NATIVE_SOURCES if src not in [
        "BlockTreeTaskPool.cc",
        "block_pool/DiskBlockIO.cc",
        "block_pool/IBlockPool.cc",
        "diagnostic/CacheDiagnostics.cc",
    ]],
    hdrs = BTC_NATIVE_HEADERS,
    includes = ["include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [
        ":block_tree_task_pool",
        ":cache_diagnostics",
        ":disk_block_io",
        ":native_block_pool",
        ":transfer_task_registry",
        "//rtp_llm/cpp/cache:block_index",
        "//rtp_llm/cpp/cache:cache_group_type",
        ":malloc_status",
        "//rtp_llm/cpp/cache:native_cache_interfaces",
        "//rtp_llm/cpp/cache/events:kv_cache_event",
        "//rtp_llm/cpp/utils:async_profiling_scope",
        "//rtp_llm/cpp/utils:error_code",
    ],
)

cc_test(
    name = "rank_worker_abi",
    srcs = ["tests/contract/RankWorkerContract.cc"],
    deps = [":btc_core"],
)

cc_test(
    name = "multi_rank_contract",
    srcs = ["tests/contract/MultiRankContract.cc"],
    deps = [":btc_core"],
)

cc_test(
    name = "native_tree_contract",
    srcs = ["tests/contract/NativeTreeContract.cc"],
    deps = [":btc_core"],
)

cc_test(
    name = "native_worker_contract",
    srcs = ["tests/contract/NativeWorkerContract.cc"],
    deps = [":btc_core"],
)

cc_test(
    name = "remote_prefetch_pressure_contract",
    srcs = ["tests/contract/RemotePrefetchPressureContract.cc"],
    deps = [":btc_core"],
)

cc_test(
    name = "block_tree_cache_contract",
    srcs = ["tests/contract/Contract.cc"],
    deps = [":btc_core"],
)

cc_library(
    name = "block_tree_cache_layout",
    srcs = ["BlockTreeCacheLayout.cc"],
    hdrs = ["BlockTreeCacheLayout.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:crc_block_copy",
    ],
)

cc_library(
    name = "storage_completion_admission",
    hdrs = ["storage_backend/StorageCompletionAdmission.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "transfer_task_registry",
    hdrs = [
        "transfer/TransferTaskRegistry.h",
    ],
    visibility = ["//visibility:public"],
    deps = [":cache_diagnostics"],
)

cc_library(
    name = "transfer_buffer_safety",
    hdrs = ["transfer/TransferBufferSafety.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Leaf pool target: DeviceBlockPool owns its backing and inherits the common
# IBlockPool lifecycle. Keeping this target below allocator/group targets avoids
# a cycle with the full block_tree_cache library.
cc_library(
    name = "native_block_pool",
    srcs = ["block_pool/IBlockPool.cc"],
    hdrs = ["block_pool/IBlockPool.h"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/cpp/cache:block_index"],
)

cc_library(
    name = "device_block_pool",
    srcs = [
        "block_pool/DeviceBlockPool.cc",
    ],
    hdrs = [
        "block_pool/DeviceBlockPool.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":native_block_pool",
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/cache:block_index",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:memory_layout_strategy",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "cache_diagnostics",
    srcs = ["diagnostic/CacheDiagnostics.cc"],
    hdrs = ["diagnostic/CacheDiagnostics.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "block_tree_task_pool",
    srcs = ["BlockTreeTaskPool.cc"],
    hdrs = ["BlockTreeTaskPool.h"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disk_block_io",
    srcs = ["block_pool/DiskBlockIO.cc"],
    hdrs = ["block_pool/DiskBlockIO.h"],
    visibility = ["//visibility:public"],
    deps = [":cache_diagnostics"],
)

cc_library(
    name = "disk_io_executor",
    srcs = ["block_pool/DiskIOExecutor.cc"],
    hdrs = ["block_pool/DiskIOExecutor.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":disk_block_io",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:thread",
    ],
)

cc_library(
    name = "crc_dump_writer",
    srcs = ["transfer/CrcDumpWriter.cc"],
    hdrs = ["transfer/CrcDumpWriter.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:crc_block_copy_hdr",
    ],
)

cc_library(
    name = "transfer_common",
    srcs = [
        "metrics/BlockTreeCacheMetricsReporter.cc",
        "metrics/CacheStageMetrics.cc",
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:group_set_sources",
        "transfer/CrcCopyExecutor.cc",
        "transfer/CrcDumpWriterMetadata.cc",
        "block_pool/DiskBlockPool.cc",
        "block_pool/DiskMountGuard.cc",
        "block_pool/AlignedHostMemory.cc",
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:host_block_pool_sources",
        "block_pool/HostStagingBlockPool.cc",
    ] + select({
        "@//:using_cuda": ["block_pool/CudaHostMemoryRegistration.cc"],
        "//conditions:default": ["block_pool/UnsupportedHostMemoryRegistration.cc"],
    }),
    hdrs = [
        "BlockTree.h",
        "metrics/BlockTreeCacheMetricsReporter.h",
        "metrics/CacheObserver.h",
        "metrics/CacheLockGuard.h",
        "metrics/CacheStageMetrics.h",
        "group_set/GroupSetResource.h",
        "evict/BlockTreeEvictor.h",
        "metrics/EvictionMetrics.h",
        "metrics/ReuseMetrics.h",
        "metrics/TransferMetrics.h",
        "group_set/GroupSet.h",
        "evict/EvictionHeap.h",
        "evict/EvictionTask.h",
        "group_set/FullGroupSet.h",
        "group_set/LinearGroupSet.h",
        "group_set/SWAGroupSet.h",
        "ScopeRollback.h",
        "TreeNode.h",
        "transfer/TransferAsyncContext.h",
        "transfer/TransferBatchAsyncContext.h",
        "transfer/FrozenTask.h",
        "transfer/TransferCompletionCleanup.h",
        "transfer/TransferCompletionTypes.h",
        "transfer/TransferStageState.h",
        "transfer/CrcCopyExecutor.h",
        "transfer/DeviceHostCopyGeometry.h",
        "transfer/TransferTypes.h",
        "transfer/TransferStageTimer.h",
        "transfer/CompletionStage.h",
        "block_pool/DiskBlockPool.h",
        "block_pool/DiskMountGuard.h",
        "block_pool/AlignedHostMemory.h",
        "block_pool/HostStagingBlockPool.h",
        "block_pool/HostMemoryRegistration.h",
        "transfer/DiskTransferUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":btc_core",
        "//rtp_llm/cpp/cache:pool_metrics",
        ":cache_diagnostics",
        ":block_tree_key",
        ":prefix_coverage",
        ":transfer_buffer_safety",
        ":transfer_task_registry",
        ":block_tree_cache_layout",
        "@havenask//aios/autil:common_macros",
        "//rtp_llm/cpp/cache/events:kv_cache_event",
        ":block_tree_task_pool",
        ":crc_dump_writer",
        ":device_block_pool",
        ":disk_block_io",
        ":disk_io_executor",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_index",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:single_type_cache_manager",
        "//rtp_llm/cpp/cache:coordinator_cache_manager_hdr",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "//rtp_llm/cpp/utils:async_profiling_scope",
        "@havenask//aios/autil:thread",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "transfer_work_loop",
    srcs = ["//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:transfer_work_loop_sources"],
    hdrs = ["transfer/TransferWorkLoop.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_common",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
    ] + torch_deps() + no_block_copy_link_deps(),
)

cc_library(
    name = "block_tree_cache",
    srcs = [
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:engine_sources",
        "transfer/BlockTransferRequestConverter.cc",
        "transfer/TransferCompletionQuery.cc",
        "transfer/TransferCompletionForget.cc",
        "transfer/FrozenTaskRecovery.cc",
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:per_rank_worker_sources",
    ],
    hdrs = [
        "BlockTreeCache.h",
        "diagnostic/FullPrefixInvariantScanner.h",
        "BlockTreeCacheFactory.h",
        "evict/EvictionTaskRunner.h",
        "load/BlockTreeLoader.h",
        "load/LoadJoinRegistry.h",
        "transfer/BlockTransferRequestConverter.h",
        "load/LoadAsyncContext.h",
        "load/LoadTaskRunner.h",
        "storage_backend/StorageBackend.h",
        "storage_backend/StorageBackendExecutor.h",
        "store/BlockTreeStorer.h",
        "transfer/BlockTransferDispatcher.h",
        "transfer/TransferCompletionQuery.h",
        "transfer/TransferCompletionForget.h",
        "transfer/FrozenTaskRecovery.h",
        "transfer/MultiRankBlockTransferEngine.h",
        "transfer/PerRankBlockTransferEngine.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":btc_core",
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:rtp_cache_diagnostics",
        "//rtp_llm/cpp/cache/block_tree_cache/integration/rtp_llm:rtp_storage_backend",
        ":cache_diagnostics",
        ":malloc_status",
        ":transfer_task_registry",
        ":block_tree_cache_layout",
        ":transfer_common",
        ":transfer_work_loop",
        "//rtp_llm/cpp/cache/events:kv_cache_event",
        ":block_tree_task_pool",
        ":crc_dump_writer",
        ":device_block_pool",
        ":disk_block_io",
        ":disk_io_executor",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:cache_capacity_negotiator",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:single_type_cache_manager",
        "//rtp_llm/cpp/cache:coordinator_cache_manager_hdr",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "//rtp_llm/cpp/utils:async_profiling_scope",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "@havenask//aios/autil:thread",
    ] + torch_deps() + no_block_copy_link_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)
