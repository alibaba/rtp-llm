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
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

cc_test(
    name = "engine_test",
    srcs = glob([
        "*EngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = [
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",        
        "//maga_transformer/cpp:engines",
        "//:gpt_init_params",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },

)
