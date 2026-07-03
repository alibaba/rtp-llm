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
    name = "full_eviction_test",
    srcs = ["FullEvictionTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "full_swa_eviction_test",
    srcs = ["FullSWAEvictionTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "full_linear_eviction_test",
    srcs = ["FullLinearEvictionTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "full_swa_linear_eviction_test",
    srcs = ["FullSWALinearEvictionTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "full_swa_non_reusable_eviction_test",
    srcs = ["FullSWANonReusableEvictionTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)
