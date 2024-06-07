load("//:def.bzl", "copts", "cuda_copts", "torch_deps")
load("//src/fastertransformer/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
] + device_linkopts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//maga_transformer/cpp:models",
    "//:gpt_init_params",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + device_impl_target()

cc_library(
    name = "test_utils",
    hdrs = [
        "ModelTestUtil.h",
    ],
    srcs = [
        "ModelTestUtil.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "gpt_model_test",
    srcs = [
        "GptModelTest.cc",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
        "//src/fastertransformer/devices/torch_impl:torch_reference_impl",
        "//src/fastertransformer/utils:utils",
    ],
    env = device_test_envs(),
)

cc_test(
    name = "sampler_test",
    srcs = [
        "SamplerTest.cc",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata"
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
