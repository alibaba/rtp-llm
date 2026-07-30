load("//:def.bzl", "copts", "cc_test_wrapper")
load("@arch_config//:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

# ExecOpsTest.cc has no CUDA/ROCm-specific symbols of its own; only the runtime
# implementations it links against are platform-specific.
exec_ops_test_common_deps = [
    "//rtp_llm/cpp/cache:cache_config",
    "//rtp_llm/cpp/cache:kv_cache_specs",
    "//rtp_llm/cpp/utils:kv_cache_utils",
    "//rtp_llm/models_py/bindings:op_defs",
    "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest_main",
] + torch_deps()

exec_ops_test_deps = exec_ops_test_common_deps + [
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
]

# Selects by fixture, so a new CacheStoreOpsTest case is gated by declaration
# rather than by matching a name prefix.
cache_store_test_filter = "CacheStoreOpsTest.*"

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

# ROCm counterpart of cache_store_op_test: same fixture, ROCm runtime provider.
# A separate rocm-tagged target rather than a select() on the CUDA one, because
# exec_properties cannot be a select() under the prelude's cc_test override.
cc_test_wrapper(
    name = "cache_store_op_rocm_test",
    srcs = ["ExecOpsTest.cc"],
    args = ["--gtest_filter=" + cache_store_test_filter],
    deps = exec_ops_test_common_deps + [
        "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
    ],
    copts = test_copts,
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
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
    exec_properties = select({
        "@//:using_cuda": {"gpu": "H20"},
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {},
    }),
)
