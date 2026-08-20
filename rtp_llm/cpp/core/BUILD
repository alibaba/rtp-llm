load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "memory_types",
    hdrs = ["MemoryTypes.h"],
    copts = copts(),
)

cc_library(
    name = "fused_copy_types",
    hdrs = ["FusedCopyTypes.h"],
    copts = copts(),
)

cc_library(
    name = "op_status",
    srcs = ["OpStatus.cc"],
    hdrs = ["OpStatus.h"],
    deps = [
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:stack_trace",
    ],
    copts = copts(),
)

cc_library(
    name = "copy_types",
    srcs = ["CopyTypes.cc"],
    hdrs = ["CopyTypes.h"],
    deps = [
        ":fused_copy_types",
        ":memory_types",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = copts(),
)

# Header-only; safe to link into every .so.  Engine targets
# (KVCacheManager etc. -> libth_transformer.so) depend on this and resolve
# runtimeCopy / runtimeBatchCopy / runtimeMaskLogits at load-time from
# librtp_compute_ops.so.  Linking the srcs into libth_transformer.so would
# give it its own thread_local `getOverlapStream()` instance (different
# CUDA stream) and produce non-deterministic numerical output on
# overlapped-copy paths (observed as golden mismatch on
# mla_cudagraph_fp8pb_deepep_dp2 dp=2).
cc_library(
    name = "copy_ops_hdrs",
    hdrs = ["CopyOps.h"],
    deps = [
        ":copy_types",
    ] + torch_deps(),
    copts = copts(),
)

# Impl + thread_local streams.  MUST live in exactly one .so (via
# th_compute_lib -> librtp_compute_ops.so). The non-CUDA/ROCm branch is only a
# host-copy compile/link fallback; RTP-LLM has no supported CPU engine backend.
cc_library(
    name = "copy_ops_impl",
    srcs = ["CopyOps.cc"],
    deps = [
        ":copy_ops_hdrs",
        ":op_status",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/runtime:device_error",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            ":split_kv_cache_copy",
            "//rtp_llm/cpp/kernels:copy_kernels",
            "//rtp_llm/cpp/kernels:fused_copy_kernel",
            "//rtp_llm/cpp/kernels:mask_logits_kernel",
            "//rtp_llm/cpp/kernels:split_kv_copy_kernel",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/cpp/kernels:copy_kernels",
            "//rtp_llm/cpp/kernels:fused_copy_kernel",
            "//rtp_llm/cpp/runtime:rocm_compat",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)

cc_library(
    name = "split_kv_cache_copy",
    srcs = select({
        "@//:using_cuda": ["SplitKvCacheCopy.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@//:using_cuda": ["SplitKvCacheCopy.h"],
        "//conditions:default": [],
    }),
    deps = select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/kernels:split_kv_copy_kernel",
            "//rtp_llm/cpp/runtime:device_error",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
)
