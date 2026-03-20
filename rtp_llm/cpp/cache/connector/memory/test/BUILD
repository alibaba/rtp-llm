load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

# memory_connector_test needs a real CUDA device (DeviceFactory::initDevices),
# but only the device registration path — not the full cuda_impl with all kernels.
# Using gpu_register instead of cuda_impl cuts the transitive dep graph dramatically
# and avoids linking the entire kernel/cutlass/flash-attention stack.
cc_test(
    name = "memory_connector_test",
    srcs = [
        "KVCacheMemoryConnectorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/devices/cuda_impl:gpu_register",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/cache/connector/memory/test/mock:mock_memory_connector_lib",
        "//rtp_llm/cpp/cache/connector/memory/test/mock:test_rpc_service_lib",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    env = {},
    exec_properties = {'gpu':'A10'},
)

# Pure logic test — no GPU or CUDA needed.
cc_test(
    name = "memory_block_cache_test",
    srcs = [
        "MemoryBlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

# gRPC-only test — no GPU or CUDA needed.
cc_test(
    name = "memory_async_context_test",
    srcs = [
        "MemoryAsyncContextTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)
