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
    "//rtp_llm/cpp/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/testing:base_tests",
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
        "//rtp_llm/cpp/cuda/ops:gpu_base",
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

cc_test(
    name = "torch_beam_search_test",
    srcs = [
        "TorchBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda/ops:torch_beam_search_op_impl",
        "@com_google_googletest//:gtest",
    ] + torch_deps(),
)

# Aliases: CudaGraph tests at //rtp_llm/cpp/cuda/CudaGraph/tests/
alias(
    name = "test_cuda_graph_runner_libs",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:test_cuda_graph_runner_libs",
    visibility = ["//visibility:public"],
)

alias(
    name = "test_cuda_graph_runner",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:test_cuda_graph_runner",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_graph_test_utils",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:cuda_graph_test_utils",
    visibility = ["//visibility:public"],
)

alias(
    name = "CudaGraphDecodePadding",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:CudaGraphDecodePadding",
    visibility = ["//visibility:public"],
)

alias(
    name = "CudaGraphPrefill",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:CudaGraphPrefill",
    visibility = ["//visibility:public"],
)

alias(
    name = "CudaGraphCopyKernelTest",
    actual = "//rtp_llm/cpp/cuda/CudaGraph/tests:CudaGraphCopyKernelTest",
    visibility = ["//visibility:public"],
)
