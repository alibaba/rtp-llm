load("//:def.bzl", "cc_test_wrapper", "copts")

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
] + cuda13_torch_link_deps

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
    deps = test_deps,
    env = {},
)

cc_test(
    name = "copy_engine_test",
    srcs = ["CopyEngineTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)
