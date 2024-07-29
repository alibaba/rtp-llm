load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-lc10_cuda",
    "-ltorch_cuda",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lnccl",
    "-lnvToolsExt",
]

cc_library(
    name = "mock_engine",
    hdrs = [
        "MockEngine.h"
    ],
    srcs = [
    ],
    copts = test_copts,
    deps =  [
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "@flash_attention//:flash_attention2_impl",
        "//maga_transformer/cpp:normal_engine",
        "//:gpt_init_params",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "batch_stream_processor_test",
    srcs = glob([
        "NormalBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = [
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "@flash_attention//:flash_attention2_impl",
        "//maga_transformer/cpp:normal_engine",
        "//:gpt_init_params",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    }
)

cc_test(
    name = "engine_test",
    srcs = glob([
        "EngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps =  [
        ":mock_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
