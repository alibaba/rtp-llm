load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "CudaGraphTestRunner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda/ops:gpu_base",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "test_cuda_graph_runner",
    deps = [
        ":test_cuda_graph_runner_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_library(
    name = "cuda_graph_test_utils",
    srcs = ["cuda_graph_test_utils.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "CudaGraphDecodePadding",
    srcs = [
        "CudaGraphDecodePadding.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        # "CUDA_LAUNCH_BLOCKING" : "1",
        # "CUDA_VISIBLE_DEVICES" : "7",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "CudaGraphPrefill",
    srcs = [
        "CudaGraphPrefill.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        # "CUDA_LAUNCH_BLOCKING" : "1",
        # "CUDA_VISIBLE_DEVICES" : "7",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "SEQ_SIZE_PER_BLOCK": "64",
        "LOAD_PYTHON_MODEL": "1",
        "ACT_TYPE": "BF16",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "CudaGraphCopyKernelTest",
    srcs = [
        "CudaGraphCopyKernelTest.py",
    ],
    data = [
        "//:th_transformer",
    ],
    deps = [],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
