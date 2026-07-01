load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "requirement", "torch_deps")

cc_test = cc_test_wrapper

requirement([
    "torch",
])

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

model_inputs_logger_deps = [
    "//rtp_llm/cpp/models:model_inputs_logger",
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
        ":torch",
    ] + select({
        "@//:using_cuda12": [
            "//rtp_llm/models_py/standalone:auto_model",
        ],
        "//conditions:default": [],
    }),
)
