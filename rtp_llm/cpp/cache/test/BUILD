load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

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

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/cache",
    "//rtp_llm/cpp/utils:core_utils",
    ":cache_config_test_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + select({
    "@//:using_rocm": [
        "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
    ],
    "//conditions:default": [
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
}) + torch_deps() + cuda13_torch_link_deps

cc_library(
    name = "mock_async_context",
    hdrs = ["MockAsyncContext.h"],
    copts = test_copts,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_async_context",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "async_context_test",
    srcs = ["AsyncContextTest.cc"],
    copts = test_copts,
    env = {},
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = [
        ":mock_async_context",
        "//rtp_llm/cpp/cache:cache_async_context",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cache_config_test_utils",
    hdrs = [
        "CacheConfigTestUtils.h",
        "TestLayoutSpec.h",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/config:model_config",
    ],
)

cc_test(
    name = "cache_topology_test",
    srcs = ["CacheTopologyTest.cc"],
    copts = test_copts,
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:cache_config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cache_layer_layout_test",
    srcs = ["CacheLayerLayoutTest.cc"],
    copts = test_copts,
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test(
    name = "kv_cache_layout_view_test",
    srcs = ["KVCacheLayoutViewTest.cc"],
    copts = test_copts,
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/models_py/bindings:op_defs",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test(
    name = "cp_slot_mapper_test",
    srcs = [
        "CPSlotMapperTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_resource_local_cache_keys_test",
    srcs = [
        "KVCacheResourceLocalCacheKeysTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cache_config_creator_py_test_lib",
    srcs = ["CacheConfigCreatorPyTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    alwayslink = True,
)

cc_binary(
    name = "cache_config_creator_py_test",
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [":cache_config_creator_py_test_lib"],
)

cc_library(
    name = "block_pool_test_helper",
    srcs = [
    ],
    hdrs = [
        "BlockPoolTestHelper.h",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
    deps = test_deps,
    alwayslink = 1,
)

cc_library(
    name = "block_tree_cache_allocator_test_helper",
    hdrs = [
        "BlockTreeCacheAllocatorTestHelper.h",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    ],
)

cc_test(
    name = "block_pool_device_malloc_test",
    srcs = [
        "BlockPoolDeviceMallocTest.cc",
    ],
    copts = test_copts,
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20_CU13"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
    tags = ["H20_CU13"],
    deps = [
        ":cache_config_test_utils",
        "//rtp_llm/cpp/cache:device_block_pool_config_helper",
        "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "memory_layout_strategy_test",
    srcs = [
        "MemoryLayoutStrategyTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps,
)

cc_test(
    name = "single_type_kv_cache_allocator_test",
    srcs = [
        "SingleTypeCoordinatorCacheManagerTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        ":block_tree_cache_allocator_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
    ],
)

cc_test(
    name = "kv_cache_manager_test",
    srcs = [
        "KVCacheManagerTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/testing:device_test_utils",
    ],
)

cc_test(
    name = "kv_cache_manager_with_tier_cache_test",
    srcs = [
        "KVCacheManagerWithTierCacheBasicTest.cc",
        "KVCacheManagerWithTierCacheCapacityTest.cc",
        "KVCacheManagerWithTierCacheComplexTopologyTest.cc",
        "KVCacheManagerWithTierCacheLoadFailureTest.cc",
        "KVCacheManagerWithTierCacheTestBase.cc",
        "KVCacheManagerWithTierCacheTestBase.h",
        "KVCacheManagerWithTierCacheWatermarkTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/normal_engine",
    ],
)

cc_test(
    name = "kv_cache_manager_cp_test",
    srcs = [
        "KVCacheManagerCPSlotMapperTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
    ],
)

cc_test(
    name = "hybrid_kv_cache_allocator_test",
    srcs = [
        "HybridTypeCoordinatorCacheManagerTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        ":block_tree_cache_allocator_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    ],
)

cc_test(
    name = "hybrid_kv_cache_allocator_cp_shard_test",
    srcs = [
        "HybridCoordinatorCacheManagerCPShardTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        ":block_tree_cache_allocator_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache",
    ],
)

cc_test(
    name = "hybrid_pool_kv_cache_allocator_test",
    srcs = [
        "HybridPoolCoordinatorCacheManagerTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        ":block_tree_cache_allocator_test_helper",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    ],
)

cc_test(
    name = "dsv4_cache_test",
    srcs = [
        "DSV4CacheTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        ":block_pool_test_helper",
        ":block_tree_cache_allocator_test_helper",
        "//rtp_llm/cpp/cache:kv_cache_transfer_planner",
        "//rtp_llm/cpp/cache/block_tree_cache",
    ],
)

cc_test(
    name = "kv_cache_resource_test",
    srcs = [
        "KVCacheResourceTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps,
)

cc_test(
    name = "cache_group_publication_test",
    srcs = [
        "CacheGroupPublicationTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {},
    deps = [
        "//rtp_llm/cpp/cache:cache_group_type",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "coordinator_cache_manager_atomic_update_test",
    srcs = ["CoordinatorCacheManagerAtomicUpdateTest.cc"],
    copts = copts(),
    env = {
        "LOG_LEVEL": "ERROR",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":block_tree_cache_allocator_test_helper",
        ":cache_config_test_utils",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:coordinator_cache_manager",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
        ],
        "//conditions:default": [
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
    }) + cuda13_torch_link_deps,
)
