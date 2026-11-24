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

test_runtime_deps = select({
    "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
    "//conditions:default": ["//rtp_llm/models_py/bindings/core:exec_ops_test_lib"],
})

test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + test_runtime_deps + cuda13_torch_link_deps + torch_deps() + no_block_copy_link_deps()

cc_test(
    name = "mock_disk_cache_source_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["MockDiskCacheSourceTest.cc"],
    copts = test_copts,
    linkopts = ["-Wl,--wrap=posix_memalign", "-Wl,--wrap=free"],
    tags = ["L20_CU13", "exclusive-if-local"],
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils"],
)

cc_test(
    name = "mock_memory_cache_source_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["MockMemoryCacheSourceTest.cc"],
    copts = test_copts,
    tags = ["L20_CU13", "exclusive-if-local"],
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
)

cc_test(
    name = "block_tree_loader_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["BlockTreeLoaderTest.cc"],
    copts = test_copts,
    tags = [
        "L20_CU13",
        "exclusive-if-local",
    ],
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "load_async_context_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["LoadAsyncContextTest.cc"],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils"],
    tags = ["L20_CU13"],
)

cc_test(
    name = "load_task_runner_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["LoadTaskRunnerTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "load_join_registry_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["LoadJoinRegistryTest.cc"],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils"],
    tags = ["L20_CU13"],
)
