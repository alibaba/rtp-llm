load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "requirement", "torch_deps")

cc_test = cc_test_wrapper

requirement([
    "torch",
])

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

model_inputs_logger_deps = [
    "//rtp_llm/cpp/models:models",
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

cc_binary(
    name = "model_inputs_logger_dump_tool",
    srcs = [
        "ModelInputsLoggerDumpTool.cc",
    ],
    copts = test_copts,
    deps = model_inputs_logger_deps,
)

py_test(
    name = "model_inputs_logger_replay_test",
    srcs = [
        "model_inputs_logger_replay_test.py",
    ],
    data = [
        ":model_inputs_logger_dump_tool",
    ],
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
        ":torch",
    ],
)
