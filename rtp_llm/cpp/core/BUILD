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
                           "//rtp_llm/cpp/rocm:rocm_types_hdr"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc"
    ],
    hdrs = [
        "allocator.h"
    ],
    copts = copts(),
    deps = [
        ":types",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:stack_trace",
        "@havenask//aios/autil:stack_tracer",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cpu_allocator",
    srcs = [
        "cpu_allocator.cc"
    ],
    hdrs = [
        "cpu_allocator.h"
    ],
    copts = copts(),
    deps = [
        ":allocator",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    include_prefix = "src",
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
    name = "event",
    hdrs = [
        "Event.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "torch_event",
    hdrs = [
        "torch_utils/TorchEvent.h",
    ],
    deps = [
        ":event",
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

# Shape checking utilities (migrated from devices/)
cc_library(
    name = "shape_check",
    hdrs = [
        "ShapeCheck.h",
    ],
    srcs = [
        "ShapeCheck.cc",
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
        ":event",
        ":common_defines",
        ":shape_check",
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

# DistributedComm — c10d ProcessGroup registry and communication ops.
# Replaces custom NCCL infrastructure with torch.distributed ProcessGroups.
cc_library(
    name = "distributed_comm_hdr",
    hdrs = [
        "DistributedComm.h",
    ],
    deps = [
        ":op_data",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "distributed_comm_srcs",
    srcs = [
        "DistributedComm.cc",
    ],
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
        ":event",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models:stats",
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

# Test-only library for ExecOps.cc (does NOT include DistributedComm.cc).
# Use this for tests that already link cuda_impl (which has DistributedComm.cc).
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
        ":devices_base",
        ":event",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":torch_event",
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
        "//rtp_llm/cpp/kernels:kernels_sampling",
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
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:config_modules",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

# ExecCtxExport.h — ExecCtxExporter base class (header-only).
# The .cc file is compiled via gpu_base filegroup (CUDA) or exec_ctx_rocm (ROCm).
# CPU/ARM builds only use the header; the methods are never called at runtime.
cc_library(
    name = "devices_base",
    hdrs = [
        "ExecCtxExport.h",
    ],
    deps = [
        ":device_data",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:config_modules",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)

filegroup(
    name = "devices_base_srcs",
    srcs = [
        "ExecCtxExport.cc",
    ],
    visibility = ["//visibility:public"],
)


filegroup(
    name = "exec_ctx_impl_srcs",
    srcs = [
        "CudaOps.cc",
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
        "DistributedComm.cc",
    ],
    visibility = ["//visibility:public"],
)

# Split into two targets so that the process-wide global state
# (globalRuntime / globalPgState) lives in exactly ONE .so.
#
# exec_ctx_state: ExecOps.cc + DistributedComm.cc
#   Defines the non-inline globalRuntime() / globalPgState() singletons.
#   Linked ONLY into librtp_compute_ops.so (via th_compute_lib).
#
# exec_ctx_ops: CudaOps.cc + sample ops + ExecCtxExport.cc
#   Platform op implementations that do NOT directly touch global state.
#   Linked into both .so files; calls to state accessors are resolved
#   at load-time from librtp_compute_ops.so.

cc_library(
    name = "exec_ctx_state",
    hdrs = [
        "ExecOps.h",
        "DistributedComm.h",
    ],
    srcs = [
        "ExecOps.cc",
        "DistributedComm.cc",
    ],
    deps = [
        ":device_data",
        ":devices_base",
        ":event",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":torch_event",
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
        "//rtp_llm/cpp/kernels:kernels_sampling",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/cpp/kernels/rocm/sampling:sampling",
            "//rtp_llm/cpp/rocm:rocm_host_utils",
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
        "ExecCtxExport.cc",
        "CudaOps.cc",
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
    ],
    deps = [
        ":exec_ops_hdr",
        ":distributed_comm_hdr",
        ":device_data",
        ":devices_base",
        ":event",
        ":op_data",
        ":type_convert",
        ":types_hdr",
        ":torch_event",
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
        "//rtp_llm/cpp/kernels:kernels_sampling",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/cpp/kernels/rocm/sampling:sampling",
            "//rtp_llm/cpp/rocm:rocm_host_utils",
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
        "DistributedComm.h",
    ],
    deps = [
        ":exec_ctx_state",
        ":exec_ctx_ops",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

