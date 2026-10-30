load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
    name = "hidden_state_capture_publisher_test",
    srcs = ["HiddenStateCapturePublisherTest.cc"],
    copts = test_copts,
    deps = test_deps + ["@local_config_python//:python_lib"],
    timeout = "moderate",
)

cc_test(
    name = "model_types_test",
    srcs = ["ModelTypesTest.cc"],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/distribute:cpu_tp_broadcaster"],
    env = {"TEST_USING_DEVICE": "CUDA"},
    timeout = "short",
)

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

cc_library(
    name = "pywrapped_model_cache_store_integration_test_lib",
    testonly = True,
    srcs = ["PyWrappedModelCacheStoreIntegrationTest.cc"],
    alwayslink = True,
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/models:context_parallel",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/core:runtime_device_hdr",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_binary(
    name = "th_pywrapped_model_cache_store_integration_test",
    testonly = True,
    deps = [":pywrapped_model_cache_store_integration_test_lib"],
    linkshared = 1,
)

py_library(
    name = "pywrapped_model_capture_fixture",
    testonly = True,
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    data = [":th_pywrapped_model_cache_store_integration_test"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "pywrapped_model_cp_capture_test",
    srcs = ["pywrapped_model_cp_capture_test.py"],
    deps = [
        ":pywrapped_model_capture_fixture",
        "//rtp_llm:torch",
        "//rtp_llm/test/utils:test_util",
    ],
    timeout = "long",
    env = {
        "GPU_COUNT": "2",
        "TEST_USING_DEVICE": "CUDA",
        "WORLD_SIZE": "2",
    },
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "2"},
)

py_test(
    name = "pywrapped_model_cache_store_integration_test",
    srcs = ["pywrapped_model_cache_store_integration_test.py"],
    data = [
        ":th_pywrapped_model_cache_store_integration_test",
        "//rtp_llm/models_py:modules",
    ],
    deps = ["//rtp_llm:torch"],
    env = {"TEST_USING_DEVICE": "CUDA"},
    tags = ["manual"],
)
