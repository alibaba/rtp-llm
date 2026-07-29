load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

exec_ops_test_deps = [
    "//rtp_llm/cpp/cache:cache_config",
    "//rtp_llm/cpp/cache:kv_cache_specs",
    "//rtp_llm/cpp/utils:kv_cache_utils",
    "//rtp_llm/models_py/bindings:op_defs",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest_main",
] + torch_deps()

cache_store_test_filter = "ExecOpsTest.testWriteCacheStore*:ExecOpsTest.testLoadContextFailureDebugInfo*"

cc_test(
    name = "exec_ops_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=-" + cache_store_test_filter],
    deps = exec_ops_test_deps,
    copts = test_copts,
    tags = ["manual"],
)

# Run focused cache-store coverage in default wildcard test gates while keeping
# the broader GPU-linked ExecOps suite manual.
cc_test(
    name = "cache_store_op_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + cache_store_test_filter],
    deps = exec_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
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
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:env_util",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
        "//conditions:default": [],
    }),
    copts = test_copts,
    # Mirror the deps select so every GPU-gated case here actually executes in
    # CI: the write() success path needs one device and the device-pinning case
    # needs two. Both labels and gpu_count are established unit-test precedents
    # (H20 x2: models_py/distributed/test; MI308X-ROCM7 x2: models_py/modules/
    # base/rocm/test). CPU-only builds stay schedulable anywhere and the GPU
    # cases skip via their build guards. select() on exec_properties is
    # configurable-attribute compatible on this repo's Bazel (6.4.0): package
    # loading and the cuda12_9 test run both pass with it in place.
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20", "gpu_count": "2"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7", "gpu_count": "2"},
        "//conditions:default": {},
    }),
)
