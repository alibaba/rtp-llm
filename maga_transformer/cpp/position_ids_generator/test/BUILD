load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

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
] + select({
    "//:using_cuda": [
        "-ltorch_cuda",
        "-lc10_cuda",
    ],
    "//:using_rocm": [
        "-ltorch_hip",
        "-lc10_hip",
    ],
    "//conditions:default": [
    ],
})

cc_test(
    name = "position_ids_generator_test",
    srcs = glob([
        "PositionIdsGeneratorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = torch_deps() + [
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/core:buffer",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "//maga_transformer/cpp:position_ids_generator",
        "//:gpt_init_params",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
