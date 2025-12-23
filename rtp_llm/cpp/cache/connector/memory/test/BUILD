load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "memory_block_cache_test",
    srcs = [
        "MemoryBlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

filegroup(
    name = "mock_files",
    srcs = glob(["mock/*"]),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "memory_connector_test",
    srcs = [
        "KVCacheMemoryConnectorTest.cc",
    ] + [
        ":mock_files",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)