load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/utils:core_utils",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/engine_base/schedulers:schedulers",
    "//rtp_llm/cpp/config:config_modules",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

async_cache_test_deps = test_deps + [
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
    "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
]

cc_test(
    name = "fifo_scheduler_test",
    srcs = [
        "FIFOSchedulerTest.cc",
    ],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "fifo_scheduler_cancel_test",
    srcs = [
        "FIFOSchedulerCancelTest.cc",
    ],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "fifo_scheduler_async_cache_test",
    srcs = [
        "FIFOSchedulerAsyncCacheTest.cc",
    ],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = async_cache_test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "gather_batch_scheduler_test",
    srcs = [
        "GatherBatchSchedulerTest.cc",
    ],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
