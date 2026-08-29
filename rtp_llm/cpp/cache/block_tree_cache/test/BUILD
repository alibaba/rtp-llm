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
    name = "block_tree_task_pool_test",
    srcs = ["BlockTreeTaskPoolTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:block_tree_task_pool",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
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
    name = "block_tree_cache_factory_test",
    testonly = True,
    srcs = ["BlockTreeCacheFactoryTest.cc"],
    copts = pool_test_copts,
    tags = ["exclusive"],
    deps = pool_test_deps + [
        ":block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/testing:device_test_utils",
    ],
    env = {},
)

cc_test(
    name = "block_tree_test",
    srcs = ["BlockTreeTest.cc"],
    copts = test_copts,
    deps = test_deps + [":block_tree_cache_test_utils"],
    env = {},
)

test_suite(
    name = "block_tree_loader_test",
    tests = ["//rtp_llm/cpp/cache/block_tree_cache/load/test:block_tree_loader_test"],
)

cc_library(
    name = "bounded_thread_test_utils",
    testonly = True,
    hdrs = ["BoundedThreadTestUtils.h"],
    visibility = ["//rtp_llm/cpp/cache/block_tree_cache:__subpackages__"],
)

cc_library(
    name = "block_tree_cache_test_utils",
    srcs = ["BlockTreeCacheTestUtils.cc"],
    hdrs = [
        "BlockTreeCacheTestUtils.h",
    ],
    copts = test_copts,
    visibility = [
        "//rtp_llm/cpp/cache/block_tree_cache:__subpackages__",
        "//rtp_llm/cpp/cache/test:__pkg__",
        "//rtp_llm/cpp/engine_base/system_prompt/test:__pkg__",
    ],
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils",
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
    name = "device_full_kv_cache_group_test",
    srcs = [
        "DeviceFullKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/test:block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "device_linear_kv_cache_group_test",
    srcs = [
        "DeviceLinearKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/test:block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "device_swa_kv_cache_group_test",
    srcs = [
        "DeviceSWAKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/test:block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "device_swa_kv_cache_group_malloc_range_test",
    srcs = [
        "DeviceSWAKVCacheGroupMallocRangeTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache:cache_types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {},
)
