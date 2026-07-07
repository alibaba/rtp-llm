load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_library(
    name = "block_tree_cache",
    srcs = [
        "BlockTree.cc",
        "BlockTreeCache.cc",
        "ComponentGroup.cc",
        "EvictionHeap.cc",
        "FullComponentGroup.cc",
        "LinearComponentGroup.cc",
        "SWAComponentGroup.cc",
        "copy_engine/CopyEngine.cc",
        "copy_engine/DiskBlockIO.cc",
        "copy_engine/DiskBlockPool.cc",
    ],
    hdrs = [
        "BlockTree.h",
        "BlockTreeCache.h",
        "ComponentGroup.h",
        "EvictionHeap.h",
        "EvictionTask.h",
        "FullComponentGroup.h",
        "LinearComponentGroup.h",
        "SWAComponentGroup.h",
        "StorageBackend.h",
        "TransferDescriptor.h",
        "TreeNode.h",
        "copy_engine/CopyEngine.h",
        "copy_engine/DiskBlockIO.h",
        "copy_engine/DiskBlockPool.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "@havenask//aios/autil:lock_free",
        "@havenask//aios/autil:thread",
    ] + torch_deps() + no_block_copy_link_deps(),
)

cc_library(
    name = "block_tree_cache_pool",
    # Sources are added incrementally as each v4 pool task lands so every commit builds.
    srcs = [
        "IBlockPool.cc",
        "host/HostBlockPool.cc",
    ],
    hdrs = [
        "IBlockPool.h",
        "host/HostBlockPool.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache:cache_types",
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
