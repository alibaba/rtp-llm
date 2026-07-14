load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
    name = "device_kv_cache_group_test",
    srcs = [
        "DeviceKVCacheGroupTest.cc",
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
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache:cache_types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "RTP_LLM_PIN_HOST_BLOCK_POOL": "0",
    },
)
