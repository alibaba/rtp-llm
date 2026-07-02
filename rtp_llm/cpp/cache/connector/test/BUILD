load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps() + cuda13_torch_link_deps

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
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
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
    exec_properties = {'gpu':'H20'},
)
