load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

cc_test(
    name = "cuda_graph_host_utils_test",
    srcs = ["cuda_graph_host_utils_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_host_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "cuda_graph_test_runner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
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
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "cuda_graph_decode_padding",
    srcs = [
        "cuda_graph_decode_padding.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "cuda_graph_prefill",
    srcs = [
        "cuda_graph_prefill.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "SEQ_SIZE_PER_BLOCK": "64",
        "ACT_TYPE": "BF16",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "cuda_graph_copy_kernel_test",
    srcs = [
        "cuda_graph_copy_kernel_test.py",
    ],
    data = [
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
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
