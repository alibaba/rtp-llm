load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps() + cuda13_torch_link_deps

cc_test(
    name = "model_data_test",
    srcs = [
        "ModelDataTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "model_inputs_logger_test",
    srcs = ["ModelInputsLoggerTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/models:model_inputs_logger",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "model_inputs_replay_test",
    srcs = ["model_inputs_replay_test.py"],
    tags = ["manual"],
    deps = ["//rtp_llm/models_py/standalone:auto_model"],
    exec_properties = {"gpu": "H20"},
)

cc_library(
    name = "pywrapped_model_cache_store_integration_test_lib",
    testonly = True,
    srcs = ["PyWrappedModelCacheStoreIntegrationTest.cc"],
    alwayslink = True,
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    ] + torch_deps(),
)

cc_binary(
    name = "th_pywrapped_model_cache_store_integration_test",
    testonly = True,
    deps = [":pywrapped_model_cache_store_integration_test_lib"],
    linkshared = 1,
)

py_test(
    name = "pywrapped_model_cache_store_integration_test",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = ["//rtp_llm:torch"],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "H20"},
)
