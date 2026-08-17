load("//:def.bzl", "copts", "cc_test_wrapper")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

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
    name = "exec_ops_test",
    srcs = ["ExecOpsTest.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    copts = test_copts,
    tags = ["manual"],
)

cc_test(
    name = "distributed_comm_test",
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["manual"],
)

cc_test_wrapper(
    name = "cache_store_async_writer_test",
    srcs = ["CacheStoreAsyncWriterTest.cpp"],
    deps = [
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
)
