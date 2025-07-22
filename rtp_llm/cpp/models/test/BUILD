load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    # "-ltorch_cpu",
    "-ltorch_python",
] + device_linkopts()

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp:normal_engine",
    "//rtp_llm/cpp:dataclass",
    "//rtp_llm/cpp:models",
    "//:gpt_init_params",
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
    exec_properties = {'gpu':'A10'},
)

cc_library(
    name = "test_cuda_graph_libs",
    srcs = glob([
        "CudaGraphDecodePaddingOp.cc"
    ]),
    hdrs = [
        "CudaGraphDecodePaddingOp.h"
    ],
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/models_py/bindings:op_defs",
        "//:th_utils"
    ],
    visibility = ["//visibility:public"],
    alwayslink = True, ## make sure pybind11 link
)

cc_binary(
    name = "test_cuda_graph_ops",
    deps = [
        ":test_cuda_graph_libs",
        # "//:th_transformer"
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "CudaGraphDecodePadding",
    srcs = [
        "CudaGraphDecodePadding.py",
    ],
    data = [
        # "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        # ":py_testdata",
        # "//rtp_llm/test/model_test/test_util:test_util"
        ":test_cuda_graph_ops"
    ],
    deps = [
        # "//rtp_llm:torch",
        # "//rtp_llm:transformers",
        "//rtp_llm/test/model_test/test_util:test_util",
        # "//rtp_llm:ops"
    ],
    # env = device_test_envs(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        # "CUDA_LAUNCH_BLOCKING" : "1",
        # "TORCH_USE_CUDA_DSA" : "1"
    },
    exec_properties = {'gpu':'A10'},
    tags = ["manual"],
)
