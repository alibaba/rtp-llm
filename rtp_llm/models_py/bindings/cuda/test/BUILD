load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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

cc_test(
    name = "dspark_logits_prepare_test",
    srcs = ["DSparkLogitsPrepareTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:dspark_logits",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "staged_memory_copy_test",
    srcs = ["StagedMemoryCopyTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "//rtp_llm/models_py/bindings/cuda:no_block_copy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "no_block_copy_test",
    srcs = ["NoBlockCopyTest.cc"],
    copts = copts(),
    env = {"GPU_COUNT": "2"},
    deps = [
        "//rtp_llm/models_py/bindings/cuda:no_block_copy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "cuda_speculative_sampling_test",
    srcs = ["CudaSpeculativeSamplingTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
        "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "H20"},
)
