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
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
)

# Run the packed-mask boundary cases against the production CUDA kernel in
# default wildcard test gates while keeping the broader ExecOps suite manual.
cc_test(
    name = "packed_mask_logits_cuda_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + packed_mask_logits_test_filter],
    deps = exec_ops_test_deps,
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
)

# Exercise the CPU fallback without initializing or allocating a GPU.
cc_test(
    name = "packed_mask_logits_cpu_fallback_test",
    srcs = ["PackedMaskLogitsCpuFallbackTest.cc"],
    deps = [
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
)

cc_test(
    name = "sample_from_probs_test",
    srcs = ["SampleFromProbsTest.cc"],
    deps = [
        # ExecOps state references the platform implementations even though
        # this test only exercises sampleFromProbs. Keep ops before state so
        # the standalone binary closes both sides of that dependency under
        # the toolchain's as-needed linking.
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = test_copts,
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_test(
    name = "distributed_comm_test",
    deps = [
        "//rtp_llm/cpp/testing:distributed_test",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
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
    copts = test_copts,
    exec_properties = select({
        "@//:using_cuda": {
            "gpu": "H20_CU13",
            "gpu_count": "2",
        },
        "@//:using_rocm": {
            "gpu": "MI308X-ROCM7",
            "gpu_count": "2",
        },
        "//conditions:default": {},
    }),
    env = {
        "GPU_COUNT": "2",
    },
    # One cross-platform target: exec_properties selects the pool per build, so
    # the tag list keeps both pools' labels addressable by filter-based lane
    # selection (ut-sm9x selects it via H20_CU13; MI308X-ROCM7 mirrors the ROCm
    # build branch). Placement is governed by exec_properties, not by tags.
    tags = [
        "H20_CU13",
        "MI308X-ROCM7",
    ],
)
