load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices:devices_base",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:gpt_init_params",
    "//rtp_llm/cpp/cache_new",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

# 旧的 BlockCache 测试保留原有依赖
block_cache_test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:gpt_init_params",
    "//rtp_llm/cpp/cache_new",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "block_cache_test",
    srcs = [
        "BlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = block_cache_test_deps,
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "block_pool_test",
    srcs = [
        "BlockPoolTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'A10'},
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
    exec_properties = {'gpu':'A10'},
)