load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "mtp_batch_stream_processor_test",
    srcs = glob([
        "MtpBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "mtp_executor_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_library(
    name = "mtp_executor_test_support",
    textual_hdrs = ["MtpExecutorTest.cc"],
)

cc_test(
    name = "mtp_executor_distributed_abort_test",
    srcs = ["MtpExecutorDistributedAbortTest.cc"],
    args = ["--gtest_filter=MtpExecutorDistributedAbortTest.*"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        ":mtp_executor_test_support",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {
        'gpu': 'H20',
        'gpu_count': '2',
    },
    size = "medium",
    timeout = "moderate",
)

cc_test(
    name = "speculative_config_validator_test",
    srcs = ["SpeculativeConfigValidatorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/engine_base:speculative_config_validator",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
