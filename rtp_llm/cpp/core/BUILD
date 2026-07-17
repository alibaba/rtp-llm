load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

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
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/models_py/bindings/core:type_convert",
        "//rtp_llm/models_py/bindings/core:common_defines",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_util",
    ] + torch_deps(),
    copts = copts(),
)

# Impl + thread_local streams.  MUST live in exactly one .so (via
# th_compute_lib -> librtp_compute_ops.so).
cc_library(
    name = "copy_ops_impl",
    srcs = ["CopyOps.cc"],
    deps = [
        ":copy_ops_hdrs",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "//rtp_llm/models_py/bindings/common/kernels:kernels_copy",
            "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/models_py/bindings/common/kernels:kernels_copy",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
            "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    alwayslink = True,
)

# Back-compat alias; header-only view.  Callers that want the impl must
# depend on :copy_ops_impl explicitly (only th_compute_lib should).
alias(
    name = "copy_ops",
    actual = ":copy_ops_hdrs",
)
