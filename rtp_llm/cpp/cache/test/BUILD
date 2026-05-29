load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda12_cudart",
    shared_library = "@pip_gpu_cuda12_9_torch_nvidia_cuda_runtime_cu12//:site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
)

cc_import(
    name = "cuda12_cublas",
    shared_library = "@pip_gpu_cuda12_9_torch_nvidia_cublas_cu12//:site-packages/nvidia/cublas/lib/libcublas.so.12",
)

cc_import(
    name = "cuda12_cublas_lt",
    shared_library = "@pip_gpu_cuda12_9_torch_nvidia_cublas_cu12//:site-packages/nvidia/cublas/lib/libcublasLt.so.12",
)

cc_import(
    name = "cuda12_cufft",
    shared_library = "@pip_gpu_cuda12_9_torch_nvidia_cufft_cu12//:site-packages/nvidia/cufft/lib/libcufft.so.11",
)

cc_import(
    name = "cuda12_cupti",
    shared_library = "@pip_gpu_cuda12_9_torch_nvidia_cuda_cupti_cu12//:site-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
)

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda12_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda12_cublas",
        ":cuda12_cublas_lt",
        ":cuda12_cudart",
        ":cuda12_cufft",
        ":cuda12_cupti",
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

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
] + torch_deps() + cuda12_torch_link_deps

block_cache_test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps() + cuda12_torch_link_deps


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
    name = "swa_kv_cache_group_malloc_range_test",
    srcs = [
        "SWAKVCacheGroupMallocRangeTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:kv_cache_group",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda12_torch_link_deps,
    env = {
        "RTP_LLM_PIN_HOST_BLOCK_POOL": "0",
    },
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
