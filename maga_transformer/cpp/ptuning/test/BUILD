load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lnccl",
    "-lnvToolsExt",
]

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    "//maga_transformer/cpp:normal_engine",
    "//maga_transformer/cpp:schedulers",
    "//maga_transformer/cpp/test:test_utils",
    "//maga_transformer/cpp/normal_engine/test:mock_engine",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "ptuning_test",
    srcs = [
        "PtuningTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)

cc_test(
    name = "ptuning_constructor_test",
    srcs = [
        "PtuningConstructorTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
