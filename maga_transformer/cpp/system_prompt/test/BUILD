load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    "//src/fastertransformer/devices/cuda_impl:cuda_impl",
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
    name = "system_prompt_test",
    srcs = [
        "SystemPromptTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "system_prompt_constructor_test",
    srcs = [
        "SystemPromptConstructorTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
