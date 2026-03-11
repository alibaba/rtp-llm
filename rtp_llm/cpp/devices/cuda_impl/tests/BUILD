load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "0",
}

cc_library(
    name = "cuda_test_util",
    hdrs = [
        "CudaTestUtils.h",
    ],
    srcs = [],
    deps = [
        "@com_google_googletest//:gtest",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/base_tests:base_tests",
    ":cuda_test_util",
] + torch_deps()


cc_test(
    name = "cuda_mask_logits_op_test",
    srcs = [
        "CudaMaskLogitsOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/cuda_impl:gpu_base",
    ],
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "CudaBeamSearchOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "CudaGraphTestRunner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/devices/cuda_impl:gpu_base",
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
    name = "CudaGraphDecodePadding",
    srcs = [
        "CudaGraphDecodePadding.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer"
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD" : "scratch",
        "NOT_USE_DEFAULT_STREAM" : "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        # "CUDA_LAUNCH_BLOCKING" : "1",
        # "CUDA_VISIBLE_DEVICES" : "7",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE" : "1",
    },
    tags = ['H20'],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "CudaGraphPrefill",
    srcs = [
        "CudaGraphPrefill.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer"
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD" : "scratch",
        # "CUDA_LAUNCH_BLOCKING" : "1",
        # "CUDA_VISIBLE_DEVICES" : "7",
        # CUDA coredump settings for debugging
        # "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" : "1",
        # "CUDA_COREDUMP_SHOW_PROGRESS" : "1",
        # "CUDA_COREDUMP_GENERATION_FLAGS" : "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
        # "CUDA_COREDUMP_FILE" : "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefill/cuda_coredump_%h.%p.%t",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE" : "1",
        "SEQ_SIZE_PER_BLOCK": "64",
        "LOAD_PYTHON_MODEL": "1",
        "ACT_TYPE": "BF16",
        # "EMBEDDING_MODEL": "1",
        # "TASK_TYPE": "DENSE_EMBEDDING",
    },
    tags = ['H20'],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "CudaGraphCopyKernelTest",
    srcs = [
        "CudaGraphCopyKernelTest.py",
    ],
    data = [
        "//:th_transformer"
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "NOT_USE_DEFAULT_STREAM" : "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        "CUDA_LAUNCH_BLOCKING" : "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE" : "1",
    },
    tags = ['H20'],
    exec_properties = {'gpu':'H20'},
)
