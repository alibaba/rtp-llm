load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
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
    name = "tp_sync_model_inputs_test_ops_lib",
    testonly = True,
    srcs = ["tp_sync_model_inputs_test_ops.cc"],
    copts = cuda_copts(),
    deps = [
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ] + torch_deps(),
    alwayslink = True,
)

cc_binary(
    name = "tp_sync_model_inputs_test_ops",
    testonly = True,
    deps = [":tp_sync_model_inputs_test_ops_lib"] + cuda13_torch_link_deps,
    linkshared = True,
)

py_test(
    name = "tp_sync_model_inputs_nccl_test",
    srcs = ["tp_sync_model_inputs_nccl_test.py"],
    data = [":tp_sync_model_inputs_test_ops"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["requires-two-gpus"],
    exec_properties = {"gpu_count": "2"},
    timeout = "long",
)
