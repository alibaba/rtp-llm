load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

pool_test_copts = [
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

pool_test_deps = [
    "//rtp_llm/cpp/cache:device_block_pool_config_helper",
    "//rtp_llm/cpp/cache/block_tree_cache",
    "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/config:static_config",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps + torch_deps()

cc_test(
    name = "host_block_pool_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["HostBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "aligned_host_memory_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["AlignedHostMemoryTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "host_staging_block_pool_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["HostStagingBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "disk_block_pool_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["DiskBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "disk_mount_guard_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["DiskMountGuardTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps,
    env = {},
)

cc_test(
    name = "i_block_pool_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["IBlockPoolTest.cc"],
    copts = pool_test_copts,
    deps = pool_test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
    env = {},
)

cc_test(
    name = "device_block_pool_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["DeviceBlockPoolTest.cc"],
    copts = pool_test_copts,
    tags = ["exclusive-if-local"],
    deps = pool_test_deps,
    env = {},
)
