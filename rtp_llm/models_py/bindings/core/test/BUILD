load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

exec_ops_test_deps = [
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/cache:cache_config",
    "//rtp_llm/models_py/bindings:op_defs",
    "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "//rtp_llm/models_py/bindings/common:common",
    "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest_main",
] + torch_deps()

cache_store_test_filter = "ExecOpsTest.testWriteCacheStore*:ExecOpsTest.testLoadContextFailureDebugInfo*"
packed_mask_logits_test_filter = "ExecOpsTest.testRuntimeApplyPackedMaskLogits*"

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
    exec_properties = {'gpu':'A10'},
)

# Run the packed-mask boundary cases against the production CUDA kernel in
# default wildcard test gates while keeping the broader ExecOps suite manual.
cc_test(
    name = "packed_mask_logits_cuda_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + packed_mask_logits_test_filter],
    deps = exec_ops_test_deps,
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)

# Exercise the CPU fallback without initializing or allocating a GPU.
cc_test(
    name = "packed_mask_logits_cpu_fallback_test",
    srcs = ["PackedMaskLogitsCpuFallbackTest.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:flashinfer_runtime",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "distributed_comm_test",
    deps = [
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["manual"],
)

cc_test_wrapper(
    name = "cache_store_async_writer_test",
    srcs = ["CacheStoreAsyncWriterTest.cpp"],
    # Own dep list rather than exec_ops_test_deps: that one is unconditionally
    # CUDA-only. cache_store_async_writer is deliberately under-linked, so each
    # platform has to supply the provider for its runtime* symbols.
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
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
    # A select needs its own CUDA branch: with only the rocm branch present, CUDA
    # falls through to "//conditions:default": {} -- empty exec_properties, which
    # is not a GPU request at all. The test then lands on a CPU-only worker and
    # dies with "libnvidia-ml.so.1: cannot open shared object file" (exit 127),
    # as it did in run 69714248.
    exec_properties = select({
        "@//:using_cuda": {"gpu": "A10"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
)
