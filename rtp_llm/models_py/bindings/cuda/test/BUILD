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

cc_test(
    name = "no_block_copy_benchmark_test",
    srcs = ["NoBlockCopyBenchmarkTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/cuda:no_block_copy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "H20"},
)
