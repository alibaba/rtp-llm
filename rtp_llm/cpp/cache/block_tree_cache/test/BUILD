load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps + torch_deps() + no_block_copy_link_deps()

cc_test(
    name = "component_group_layout_test",
    srcs = ["ComponentGroupLayoutTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_library(
    name = "copy_engine_test_utils",
    srcs = ["CopyEngineTestUtils.cc"],
    hdrs = ["CopyEngineTestUtils.h"],
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "copy_engine_device_host_test",
    testonly = True,
    srcs = ["CopyEngineDeviceHostTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":copy_engine_test_utils"],
    env = {},
)

cc_test(
    name = "copy_engine_host_disk_test",
    testonly = True,
    srcs = ["CopyEngineHostDiskTest.cc"],
    copts = test_copts,
    deps = test_deps + [":copy_engine_test_utils"],
    env = {},
)

test_suite(
    name = "copy_engine_test",
    tests = [
        ":copy_engine_device_host_test",
        ":copy_engine_host_disk_test",
    ],
)

pool_test_copts = [
    "-fno-access-control",
] + copts()

pool_test_deps = [
    "//rtp_llm/cpp/cache:device_block_pool_config_helper",
    "//rtp_llm/cpp/cache/block_tree_cache",
    "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/config:static_config",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps + torch_deps()

cc_test(
    name = "i_block_pool_test",
    srcs = ["IBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "host_block_pool_test",
    srcs = ["HostBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "block_tree_cache_factory_test",
    testonly = True,
    srcs = ["BlockTreeCacheFactoryTest.cc"],
    copts = pool_test_copts,
    tags = ["exclusive"],
    deps = pool_test_deps + [
        ":block_tree_cache_test_utils",
        ":copy_engine_test_utils",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/testing:device_test_utils",
    ],
    env = {},
)

cc_test(
    name = "disk_block_pool_test",
    srcs = ["DiskBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "disk_mount_guard_test",
    srcs = ["DiskMountGuardTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "device_block_pool_test",
    srcs = ["DeviceBlockPoolTest.cc"],
    copts = pool_test_copts,
    tags = ["exclusive"],
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "block_tree_test",
    srcs = ["BlockTreeTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "eviction_heap_test",
    srcs = ["EvictionHeapTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "block_tree_evictor_test",
    srcs = ["BlockTreeEvictorTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps,
    env = {},
)

cc_test(
    name = "full_component_group_test",
    srcs = ["FullComponentGroupTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "swa_component_group_test",
    srcs = ["SWAComponentGroupTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "linear_component_group_test",
    srcs = ["LinearComponentGroupTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":block_tree_cache_test_utils"],
    env = {},
)

cc_library(
    name = "block_tree_cache_test_utils",
    srcs = ["BlockTreeCacheTestUtils.cc"],
    hdrs = [
        "BlockTreeCacheTestUtil.h",
        "BlockTreeCacheTestUtils.h",
    ],
    copts = test_copts,
    visibility = [
        "//rtp_llm/cpp/cache/block_tree_cache/test:__subpackages__",
        "//rtp_llm/cpp/cache/test:__pkg__",
        "//rtp_llm/cpp/engine_base/system_prompt/test:__pkg__",
    ],
    deps = [
        ":copy_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
        "@com_google_googletest//:gtest",
    ] + torch_deps(),
)

cc_test(
    name = "block_tree_cache_test",
    testonly = True,
    srcs = ["BlockTreeCacheTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [
        ":block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache:cache_async_context",
    ],
    env = {},
)

cc_test(
    name = "block_tree_cache_integration_test",
    testonly = True,
    srcs = ["BlockTreeCacheIntegrationTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "block_tree_cache_broadcast_test",
    testonly = True,
    srcs = ["BlockTreeCacheBroadcastTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [
        ":block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
    ],
    env = {},
)

cc_test(
    name = "block_tree_transfer_converter_test",
    srcs = ["BlockTreeTransferConverterTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)
