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
    "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
    "//rtp_llm/cpp/cache:cache_async_context",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps + torch_deps() + no_block_copy_link_deps()

cc_test(
    name = "block_tree_storer_test",
    exec_properties = {"gpu": "A10"},
    testonly = True,
    srcs = ["BlockTreeStorerTest.cc"],
    copts = test_copts,
    tags = ["exclusive-if-local"],
    deps = test_deps,
    env = {},
)

cc_test(
    name = "store_task_runner_test",
    exec_properties = {"gpu": "A10"},
    testonly = True,
    srcs = ["StoreTaskRunnerTest.cc"],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils"],
    env = {},
)
