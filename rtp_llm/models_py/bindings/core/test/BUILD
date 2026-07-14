load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

exec_ops_test_deps = [
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "//rtp_llm/models_py/bindings/common:common",
    "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest_main",
] + torch_deps()

cc_test(
    name = "exec_ops_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=-ExecOpsTest.testWriteCacheStore*"],
    deps = exec_ops_test_deps,
    copts = test_copts,
    tags = ["manual"],
)

# Cache-store address/stride/group-id coverage is part of the default cuda12_9
# gate. Keep the broader ExecOps suite manual while running these focused cases.
cc_test(
    name = "cache_store_op_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=ExecOpsTest.testWriteCacheStore*"],
    deps = exec_ops_test_deps,
    copts = test_copts,
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
