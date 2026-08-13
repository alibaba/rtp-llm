load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_test = cc_test_wrapper

test_deps = [
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/config:model_config",
    "//rtp_llm/cpp/utils:core_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps() + no_block_copy_link_deps()

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "memory_connector_test",
    srcs = [
        "KVCacheMemoryConnectorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/cache/connector/memory/test/mock:mock_memory_connector_lib",
        "//rtp_llm/cpp/cache/connector/memory/test/mock:test_rpc_service_lib",
    ],
    linkopts = [
        "-Wl,--wrap=_ZN7rtp_llm15execNoBlockCopyERKNS_15MultiCopyParamsE",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "memory_block_cache_test",
    srcs = [
        "MemoryBlockCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache:block_pool",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "memory_async_context_test",
    srcs = [
        "MemoryAsyncContextTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_async_context",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_copy_fence_test",
    srcs = ["MemoryCopyFenceTest.cc"],
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_copy_fence",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_copy_operation_id_test",
    srcs = ["MemoryCopyOperationIdTest.cc"],
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_copy_operation_id",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_copy_deadline_test",
    srcs = ["MemoryCopyDeadlineTest.cc"],
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_copy_deadline",
        "//rtp_llm/cpp/cache/connector/memory:memory_copy_fence",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_copy_task_guard_test",
    srcs = ["MemoryCopyTaskGuardTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_copy_task_guard",
        "//rtp_llm/cpp/model_rpc:remote_load_lease_retainer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:lock_free",
    ],
)
