load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "cc_test_wrapper", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "generic_sampler_batch_fallback_test",
    srcs = ["GenericSamplerBatchFallbackTest.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:generic_sampler_fallback_test_lib",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    tags = ["cpu"],
    target_compatible_with = select({
        "@//:using_cpu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

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
    copts = test_copts,
    tags = ["manual"],
    deps = exec_ops_test_deps,
)

# Run focused cache-store coverage in default wildcard test gates while keeping
# the broader GPU-linked ExecOps suite manual.
cc_test(
    name = "cache_store_op_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + cache_store_test_filter],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = exec_ops_test_deps,
)

# Run the packed-mask boundary cases against the production CUDA kernel in
# default wildcard test gates while keeping the broader ExecOps suite manual.
cc_test(
    name = "packed_mask_logits_cuda_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + packed_mask_logits_test_filter],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = exec_ops_test_deps,
)

# Exercise the CPU fallback without initializing or allocating a GPU.
cc_test(
    name = "packed_mask_logits_cpu_fallback_test",
    srcs = ["PackedMaskLogitsCpuFallbackTest.cc"],
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:flashinfer_runtime",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test(
    name = "sample_from_probs_test",
    srcs = ["SampleFromProbsTest.cc"],
    copts = test_copts,
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        # ExecOps state references the platform implementations even though
        # this test only exercises sampleFromProbs. Keep ops before state so
        # the standalone binary closes both sides of that dependency under
        # the toolchain's as-needed linking.
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test(
    name = "distributed_comm_test",
    copts = test_copts,
    tags = ["manual"],
    deps = [
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_test_wrapper(
    name = "cache_store_async_writer_test",
    srcs = ["CacheStoreAsyncWriterTest.cpp"],
    copts = test_copts,
    exec_properties = select({
        "@//:using_cuda": {"gpu": "A10_CU13"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
    # A select needs its own CUDA branch: with only the rocm branch present, CUDA
    # falls through to "//conditions:default": {} -- empty exec_properties, which
    # is not a GPU request at all. The test then lands on a CPU-only worker and
    # dies with "libnvidia-ml.so.1: cannot open shared object file" (exit 127),
    # as it did in run 69714248.
    tags = ["A10_CU13"],
    # Own dep list rather than exec_ops_test_deps: that one is unconditionally
    # CUDA-only. cache_store_async_writer is deliberately under-linked, so each
    # platform has to supply the provider for its runtime* symbols.
    deps = [
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:coordinator_cache_manager",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        ],
        "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
        "//conditions:default": [],
    }),
)
