load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/cache",
    "//rtp_llm/cpp/utils:core_utils",
    ":cache_config_test_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

block_cache_test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()


cc_library(
    name = "cache_config_test_utils",
    hdrs = [
        "CacheConfigTestUtils.h",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "block_pool_test_helper",
    hdrs = [
        "BlockPoolTestHelper.h"
    ],
    srcs = [
    ],
    copts = test_copts,
    deps = test_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "block_cache_test",
    srcs = [
        "BlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = block_cache_test_deps,
    env = {},
)

cc_test(
    name = "block_pool_test",
    srcs = [
        "BlockPoolTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "memory_layout_strategy_test",
    srcs = [
        "MemoryLayoutStrategyTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "full_kv_cache_group_test",
    srcs = [
        "FullKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "single_type_kv_cache_allocator_test",
    srcs = [
        "SingleTypeKVCacheAllocatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "kv_cache_manager_test",
    srcs = [
        "KVCacheManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
        "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
        "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "linear_kv_cache_group_test",
    srcs = [
        "LinearKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "hybrid_kv_cache_allocator_test",
    srcs = [
        "HybridKVCacheAllocatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)