load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "cache_manager_test",
    srcs = [
        "CacheManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":mock_dist_kvcache",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cache_config_creator_test",
    srcs = [
        "CacheConfigCreatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "dist_storage_3fs_test",
    srcs = [
        "DistStorage3FSTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        ":mock_dist_storage_3fs_file",
    ],
    tags = ["manual"],
)

cc_library(
    name = "mock_dist_kvcache",
    hdrs = [
        "mock/MockDistKvCache.h",
    ],
)

cc_test(
    name = "kv_cache_allocator_test",
    srcs = [
        "KVCacheAllocatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
cc_library(
    name = "mock_dist_storage_3fs_file",
    hdrs = [
        "mock/MockDistStorage3FSFile.h",
    ],
)

cc_test(
    name = "block_ref_counter_test",
    srcs = [
        "BlockRefCounterTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "block_lru_cache_test",
    srcs = [
        "BlockLRUCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "memory_block_cache_test",
    srcs = [
        "MemoryBlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)