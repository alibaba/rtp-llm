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

scenario_test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache",
    "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps

cc_test(
    name = "eviction_heap_test",
    srcs = ["EvictionHeapTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "block_tree_evictor_test",
    srcs = ["BlockTreeEvictorTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "eviction_task_runner_test",
    srcs = ["EvictionTaskRunnerTest.cc"],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
)

cc_test(
    name = "full_eviction_test",
    testonly = True,
    srcs = ["FullEvictionTest.cc"],
    copts = test_copts,
    deps = scenario_test_deps,
    env = {},
)

cc_test(
    name = "full_swa_eviction_test",
    testonly = True,
    srcs = ["FullSWAEvictionTest.cc"],
    copts = test_copts,
    deps = scenario_test_deps,
    env = {},
)

cc_test(
    name = "full_linear_eviction_test",
    testonly = True,
    srcs = ["FullLinearEvictionTest.cc"],
    copts = test_copts,
    deps = scenario_test_deps,
    env = {},
)

cc_test(
    name = "full_swa_linear_eviction_test",
    testonly = True,
    srcs = ["FullSWALinearEvictionTest.cc"],
    copts = test_copts,
    deps = scenario_test_deps,
    env = {},
)
