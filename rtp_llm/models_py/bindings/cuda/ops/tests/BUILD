load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

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

cc_library(
    name = "cuda_test_util",
    srcs = [],
    hdrs = [
        "CudaTestUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
        "@local_config_cuda//cuda:cuda_headers",
    ],
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
    copts = copts(),
    data = [],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = test_deps,
)

cc_test(
    name = "cuda_mask_logits_op_test",
    srcs = [
        "CudaMaskLogitsOpTest.cc",
    ],
    copts = copts(),
    data = [],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    ],
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    copts = copts(),
    data = [],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        "//3rdparty/flashinfer",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling",
    ],
)

cc_test(
    name = "rejection_sampling_op_test",
    srcs = [
        "CudaRejectionSamplingOpTest.cc",
    ],
    copts = copts(),
    data = [],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/cpp/testing:rejection_sampling_test_utils",
    ] + torch_deps(),
)

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "CudaBeamSearchOpTest.cc",
    ],
    copts = copts(),
    data = [],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        "//3rdparty/flashinfer",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling",
    ] + cuda13_torch_link_deps,
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
