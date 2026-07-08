load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

# Leaf pool target: the IBlockPool base plus the device (GPU) pool. Deliberately narrow so
# the allocator-owned cache targets (kv_cache_group, kv_cache_allocator_hdr,
# shared_block_cache) can depend on DeviceBlockPool without a cycle - the full
# block_tree_cache library depends back on those cache targets.
cc_library(
    name = "device_block_pool",
    srcs = [
        "DeviceBlockPool.cc",
        "IBlockPool.cc",
    ],
    hdrs = [
        "DeviceBlockPool.h",
        "IBlockPool.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:block_pool",
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
    name = "block_tree_cache",
    srcs = [
        "BlockTree.cc",
        "BlockTreeCache.cc",
        "BlockTreeCacheFactory.cc",
        "BlockTreeEvictor.cc",
        "ComponentGroup.cc",
        "EvictionHeap.cc",
        "FullComponentGroup.cc",
        "LinearComponentGroup.cc",
        "SWAComponentGroup.cc",
        "copy_engine/CopyEngine.cc",
        "host/DiskBlockIO.cc",
        "host/DiskBlockPool.cc",
        "host/DiskMountGuard.cc",
        "host/HostBlockPool.cc",
    ],
    hdrs = [
        "BlockTree.h",
        "BlockTreeCache.h",
        "BlockTreeCacheFactory.h",
        "BlockTreeEvictor.h",
        "ComponentGroup.h",
        "EvictionHeap.h",
        "EvictionTask.h",
        "FullComponentGroup.h",
        "LinearComponentGroup.h",
        "SWAComponentGroup.h",
        "StorageBackend.h",
        "TreeNode.h",
        "copy_engine/CopyEngine.h",
        "copy_engine/TransferTypes.h",
        "host/DiskBlockIO.h",
        "host/DiskBlockPool.h",
        "host/DiskMountGuard.h",
        "host/HostBlockPool.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":device_block_pool",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache:kv_cache_allocator_hdr",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "@havenask//aios/autil:lock_free",
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
