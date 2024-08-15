load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    "//maga_transformer/cpp:dataclass",
    "//maga_transformer/cpp:lora",
    "//:gpt_init_params",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda",
    "@local_config_cuda//cuda:cudart",
    "//maga_transformer/cpp:normal_engine",
] + torch_deps()

cc_test(
    name = "lora_weights_test",
    srcs = [
        "LoraWeightsTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)

cc_test(
    name = "lora_model_test",
    srcs = [
        "LoraModelTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)

cc_test(
    name = "lora_manager_test",
    srcs = [
        "LoraManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)

cc_test(
    name = "lora_norm_engine_test",
    srcs = [
        "LoraNormEngineTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)