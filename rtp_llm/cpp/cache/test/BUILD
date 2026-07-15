load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
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
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
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
        "//rtp_llm/cpp/config:model_config",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "cache_topology_test",
    srcs = ["CacheTopologyTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cache_config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

cc_test(
    name = "cache_layer_layout_test",
    srcs = ["CacheLayerLayoutTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    env = {},
)

cc_test(
    name = "kv_cache_layout_view_test",
    srcs = ["KVCacheLayoutViewTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/models_py/bindings:op_defs",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    env = {},
)

cc_test(
    name = "cp_slot_mapper_test",
    srcs = [
        "CPSlotMapperTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

cc_test(
    name = "kv_cache_resource_local_cache_keys_test",
    srcs = [
        "KVCacheResourceLocalCacheKeysTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

cc_library(
    name = "cache_config_creator_py_test_lib",
    srcs = ["CacheConfigCreatorPyTest.cc"],
    alwayslink = True,
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
)

cc_binary(
    name = "cache_config_creator_py_test",
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [":cache_config_creator_py_test_lib"],
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
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "shared_block_cache_test",
    srcs = [
        "SharedBlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = block_cache_test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "kv_cache_manager_cp_test",
    srcs = [
        "KVCacheManagerCPSlotMapperTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
        "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "hybrid_kv_cache_allocator_test",
    srcs = [
        "HybridTypeKVCacheAllocatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "hybrid_kv_cache_allocator_cp_shard_test",
    srcs = [
        "HybridKVCacheAllocatorCPShardTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "hybrid_pool_kv_cache_allocator_test",
    srcs = [
        "HybridPoolKVCacheAllocatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "swa_kv_cache_group_test",
    srcs = [
        "SWAKVCacheGroupTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "dsv4_cache_test",
    srcs = [
        "DSV4CacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":block_pool_test_helper",
        "//rtp_llm/cpp/cache:kv_cache_transfer_planner",
        "//rtp_llm/cpp/cache/connector/p2p:layer_block_converter",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "kv_cache_resource_test",
    srcs = [
        "KVCacheResourceTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
)
