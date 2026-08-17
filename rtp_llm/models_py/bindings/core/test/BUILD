load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

exec_ops_test_deps = [
    "//rtp_llm/cpp/cache:cache_config",
    "//rtp_llm/models_py/bindings:op_defs",
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

# Host-side memory-growth math assertions.
cc_test(
    name = "memory_diagnostics_test",
    srcs = ["MemoryDiagnosticsTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/models_py/bindings/core:device_data",
        "@com_google_googletest//:gtest_main",
    ],
)

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

# Run the packed-mask boundary cases against the production CUDA kernel in
# default wildcard test gates while keeping the broader ExecOps suite manual.
cc_test(
    name = "packed_mask_logits_cuda_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + packed_mask_logits_test_filter],
    deps = exec_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
)

# Production trace-memory entry points (setTraceMemory and the getGpuExecStatus baseline
# plumbing) on the linked ExecOps.cc. Deliberately NOT
# manual: memory_diagnostics_test only covers the pure growth calculation, and exec_ops_test
# (manual) never runs in default gates, so without this target the functions NormalEngine calls
# have no gated coverage.
#
# Known gap: CUDA only. Note what the gap is NOT -- on ROCm sizing does not degrade because the
# growth breakdown reads zero; NormalEngine gates the entire warmup behind #if USING_CUDA, so ROCm
# never produces a WarmUpResult and sizing takes the no-warmup formula for that reason alone. The
# platform-independent evidence for the nullopt/no-warmup formula is the no-result cases in
# MemoryEvaluationHelperTest. ZeroMeasurementDegradesToTheNoWarmupFormulaExactly separately covers
# a CUDA-capable role that produced a trusted WarmUpResult whose zero growth marks a broken trace.
# The actual uncovered code is the ROCm branch of getGpuExecStatus itself: its hipMemGetInfo device-
# total sampling and its ROCM_CHECK failure path. A ROCm target was attempted and removed:
# exec_ops_test_lib is what compiles ExecOps.cc but it needs the platform kernel providers to link
# (sampleGreedy, multiMergeCopy come from cuda/ops:cuda_impl + gpu_base on CUDA), so the ROCm
# equivalent cannot be assembled -- let alone verified -- without a ROCm toolchain. Add it as a
# select()-based single target once one is available.
cc_test(
    name = "trace_memory_ops_test",
    srcs = ["TraceMemoryOpsTest.cc"],
    deps = exec_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "H20"},
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
    # Own dep list rather than exec_ops_test_deps: that one is unconditionally
    # CUDA-only. cache_store_async_writer is deliberately under-linked, so each
    # platform has to supply the provider for its runtime* symbols.
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_config",
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
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
)
