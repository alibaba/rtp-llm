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
    name = "full_component_group_test",
    srcs = ["FullComponentGroupTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "swa_component_group_test",
    srcs = ["SWAComponentGroupTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "linear_component_group_test",
    srcs = ["LinearComponentGroupTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "block_tree_cache_test",
    srcs = ["BlockTreeCacheTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    ],
    env = {},
)

cc_test(
    name = "copy_engine_test",
    srcs = ["CopyEngineTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {"gpu": "H20"},
)

pool_test_copts = [
    "-fno-access-control",
] + copts()

pool_test_deps = [
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
    name = "disk_block_pool_test",
    srcs = ["DiskBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "device_block_pool_test",
    srcs = ["DeviceBlockPoolTest.cc"],
    copts = pool_test_copts,
    # Uses only the lightweight cache_config_test_utils (already in pool_test_deps) +
    # the merged block_tree_cache pool sources; deliberately avoids
    # block_pool_test_helper, which pulls the heavy //rtp_llm/cpp/cache monolith.
    deps = pool_test_deps,
    env = {},
    exec_properties = {"gpu": "H20"},
)
