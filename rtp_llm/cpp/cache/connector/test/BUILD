load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

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

cc_test(
    name = "coordinator_test",
    srcs = [
        "KVCacheConnectorCoordinatorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector:connector",
        "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
        "//rtp_llm/cpp/cache/connector/memory/test/mock:mock_memory_connector_lib",
        "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
    ],
    env = {},
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "async_context_test",
    srcs = [
        "AsyncContextTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector:connector",
        "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
    ],
    env = {},
)
