load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

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
] + torch_deps()

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

# Thin pybind bridge that dynamically resolves tpSyncModelInputs from the
# shipped libth_transformer.so. It deliberately does not define the production
# symbol or link CpuBroadcast.cc.
cc_binary(
    name = "tp_sync_model_inputs_production_test_bridge",
    srcs = ["TpSyncModelInputsProductionTestBridge.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/models:model_types",
    ] + torch_deps(),
    linkopts = ["-ldl"],
    linkshared = 1,
)

# Non-manual, real two-rank production-boundary coverage for the default UDS
# path and both c10d fallback paths. Keep this out of open_skip: it guards the
# Python bootstrap -> pybind -> cross-DSO tpSyncModelInputs contract.
py_test(
    name = "tp_sync_model_inputs_production_test",
    srcs = ["tp_sync_model_inputs_production_test.py"],
    data = [
        ":tp_sync_model_inputs_production_test_bridge",
        "//:rtp_compute_ops",
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    env = {
        "GPU_COUNT": "2",
        "NCCL_DEBUG": "WARN",
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "2",
    },
    tags = ["H20"],
    timeout = "moderate",
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
