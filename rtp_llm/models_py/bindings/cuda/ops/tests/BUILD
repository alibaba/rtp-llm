load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

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
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
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
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    ],
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "CudaBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "torch_beam_search_test",
    srcs = [
        "TorchBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/ops:torch_beam_search_op_impl",
        "@com_google_googletest//:gtest",
    ] + torch_deps(),
)

# Aliases: CudaGraph tests at //rtp_llm/cpp/cuda_graph/tests/
alias(
    name = "test_cuda_graph_runner_libs",
    actual = "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner_libs",
    visibility = ["//visibility:public"],
)

alias(
    name = "test_cuda_graph_runner",
    actual = "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_graph_test_utils",
    actual = "//rtp_llm/cpp/cuda_graph/tests:cuda_graph_test_utils",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_graph_decode_padding",
    actual = "//rtp_llm/cpp/cuda_graph/tests:cuda_graph_decode_padding",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_graph_prefill",
    actual = "//rtp_llm/cpp/cuda_graph/tests:cuda_graph_prefill",
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_graph_copy_kernel_test",
    actual = "//rtp_llm/cpp/cuda_graph/tests:cuda_graph_copy_kernel_test",
    visibility = ["//visibility:public"],
)
