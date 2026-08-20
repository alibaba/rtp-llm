load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

runtime_ops_test_deps = [
    "//rtp_llm/cpp/cache:cache_config",
    "//rtp_llm/cpp/cache:cache_store_writer_hdrs",
    "//rtp_llm/cpp/core:copy_ops_hdrs",
    "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
    "//rtp_llm/models_py/bindings:op_defs",
    "//rtp_llm/models_py/bindings/core:device_ops_test_impls",
    "//rtp_llm/models_py/bindings/common:common",
    "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + cuda13_torch_link_deps

cache_store_test_filter = "RuntimeOpsTest.testWriteCacheStore*:RuntimeOpsTest.testLoadContextFailureDebugInfo*"
packed_mask_logits_test_filter = "RuntimeOpsTest.testRuntimeApplyPackedMaskLogits*"

cc_test(
    name = "runtime_ops_test",
    srcs = ["RuntimeOpsTest.cc"],
    args = ["--gtest_filter=-" + cache_store_test_filter],
    deps = runtime_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "cache_store_op_test",
    srcs = ["RuntimeOpsTest.cc"],
    args = ["--gtest_filter=" + cache_store_test_filter],
    deps = runtime_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "packed_mask_logits_cuda_test",
    srcs = ["RuntimeOpsTest.cc"],
    args = ["--gtest_filter=" + packed_mask_logits_test_filter],
    deps = runtime_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "packed_mask_logits_cpu_fallback_test",
    srcs = ["PackedMaskLogitsCpuFallbackTest.cc"],
    deps = [
        "//rtp_llm/cpp/core:copy_ops_hdrs",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "distributed_comm_test",
    deps = [
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["manual"],
)

cc_test_wrapper(
    name = "cache_store_async_writer_test",
    srcs = ["CacheStoreAsyncWriterTest.cpp"],
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
)
