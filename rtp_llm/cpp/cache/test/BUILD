load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:gpt_init_params",
    "//rtp_llm/cpp/cache",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

# 旧的 BlockCache 测试保留原有依赖
block_cache_test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:gpt_init_params",
    "//rtp_llm/cpp/cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()


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
    exec_properties = {'gpu':'A10'},
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
    name = "hybrid_layer_kv_cache_allocator_test",
    srcs = [
        "HybridLayerKVCacheAllocatorTest.cc",
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
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "tp_broadcast_manager_test",
    srcs = [
        "TpBroadcastManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "@havenask//aios/autil:net",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)