load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load(
    "//bazel:device_defs.bzl",
    "cuda_test_envs",
    "device_impl_target",
    "device_runtime_deps",
    "rocm_test_envs",
)

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

# This test stays CUDA-only because it directly exercises the CUDA op
# implementation. The executor regression below has dedicated CUDA/ROCm targets.

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

mtp_executor_test_srcs = glob([
    "MtpExecutorTest.cc",
])

mtp_executor_test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/cache:cache",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/normal_engine/test:mock_engine",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + device_runtime_deps() + device_impl_target()

cc_test(
    name = "mtp_executor_test",
    srcs = mtp_executor_test_srcs,
    data = [],
    copts = test_copts,
    deps = mtp_executor_test_deps,
    env = cuda_test_envs(),
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "mtp_executor_rocm_test",
    srcs = mtp_executor_test_srcs,
    data = [],
    copts = test_copts,
    deps = mtp_executor_test_deps,
    env = rocm_test_envs(),
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
)
