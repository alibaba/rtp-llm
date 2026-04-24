load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "types_hdr",
    hdrs = [
        "Types.h",
        "Dispatch.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    srcs = [
        "Types.cc",
    ],
    deps = [
        ":types_hdr",
    ] + select({
        "@//:using_cuda": ["@local_config_cuda//cuda:cuda_headers",
                           "@local_config_cuda//cuda:cudart"],
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers",
                           "@local_config_rocm//rocm:rocm",
                           "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "type_convert",
    hdrs = [
        "torch_utils/TypeConvert.h",
    ],
    deps = [
        ":types",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common_defines",
    hdrs = [
        "CommonDefines.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "op_data",
    hdrs = [
        "OpData.h",
    ],
    srcs = [
        "OpData.cc",
    ],
    deps = [
        ":type_convert",
        ":common_defines",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/utils:stack_trace",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

# ExecOps — unified public header for runtime operations.
# Replaces the former ExecutionContext.h (now deleted).
cc_library(
    name = "exec_ops_hdr",
    hdrs = [
        "ExecOps.h",
    ],
    deps = [
        ":op_data",
        ":device_data",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_util",
    ] + torch_deps() + select({
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "exec_ops_srcs",
    srcs = [
        "ExecOps.cc",
    ],
    visibility = ["//visibility:public"],
)

# Test-only library for ExecOps.cc.
# Use this for tests that already link cuda_impl.
cc_library(
    name = "exec_ops_test_lib",
    hdrs = [
        "ExecOps.h",
    ],
    srcs = [
        "ExecOps.cc",
    ],
    deps = [
        ":device_data",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":common_defines",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

# DeviceData.h - standalone header for DeviceInitParams, DeviceType, etc.
cc_library(
    name = "device_data",
    hdrs = [
        "DeviceData.h",
    ],
    deps = [
        ":common_defines",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:config_modules",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)


filegroup(
    name = "exec_ctx_impl_srcs",
    srcs = [
        "CudaOps.cc",
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
    ],
    visibility = ["//visibility:public"],
)

# Split into two targets so that the process-wide global state
# lives in exactly ONE .so.
#
# exec_ctx_state: ExecOps.cc
#   Defines the non-inline globalRuntime() singleton + comm op callbacks.
#   Linked ONLY into librtp_compute_ops.so (via th_compute_lib).
#
# exec_ctx_ops: CudaOps.cc + sample ops
#   Platform op implementations that do NOT directly touch global state.
#   Linked into both .so files; calls to state accessors are resolved
#   at load-time from librtp_compute_ops.so.

cc_library(
    name = "exec_ctx_state",
    hdrs = [
        "ExecOps.h",
    ],
    srcs = [
        "ExecOps.cc",
    ],
    deps = [
        ":device_data",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":common_defines",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "exec_ctx_ops",
    srcs = [
        "CudaOps.cc",
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
    ],
    deps = [
        ":exec_ops_hdr",
        ":device_data",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":common_defines",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/rocm:speculative_sampling",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

# Combined target for backward compatibility — pulls in both state + ops.
cc_library(
    name = "exec_ctx_rocm",
    hdrs = [
        "ExecOps.h",
    ],
    deps = [
        ":exec_ctx_state",
        ":exec_ctx_ops",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_store_async_writer",
    hdrs = ["CacheStoreAsyncWriter.h"],
    srcs = ["CacheStoreAsyncWriter.cc"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:lock_free",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

