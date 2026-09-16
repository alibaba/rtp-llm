load("//:def.bzl", "cc_test_wrapper", "copts")
load("@rtp_llm//bazel:defs.bzl", "torch_deps")

cc_test = cc_test_wrapper

alias(
    name = "cuda13_torch_nvshmem",
    actual = "@torch//:torch",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

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
    name = "batch_copy_test",
    srcs = [
        "CudaBatchCopyTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'H20'},
)

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
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps + [
        "//3rdparty/flashinfer:flashinfer",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
    ],
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "rejection_sampling_op_test",
    srcs = [
        "CudaRejectionSamplingOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/testing:rejection_sampling_test_utils",
    ] + torch_deps(),
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "CudaBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps + [
        "//3rdparty/flashinfer:flashinfer",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
    ] + cuda13_torch_link_deps,
    exec_properties = {'gpu':'A10'},
)

# Aliases: compiled CudaGraph test binding at //rtp_llm/cpp/cuda_graph/tests/.
# Its Python drivers run under pytest/uv rather than Bazel py_test.
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
