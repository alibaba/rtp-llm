load("//:def.bzl", "rpm_library", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

# Stub weight preprocess declarations; safe to link into every .so.
cc_library(
    name = "weight_preprocess_hdrs",
    hdrs = ["WeightPreprocess.h"],
    deps = torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Stateless compatibility implementation. Keep it in librtp_compute_ops.so so
# exec_ops_hdr remains an interface-only dependency bundle.
cc_library(
    name = "weight_preprocess_impl",
    srcs = ["WeightPreprocess.cc"],
    deps = [":weight_preprocess_hdrs"] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//rtp_llm/cpp/pybind:__pkg__"],
)

alias(
    name = "weight_preprocess",
    actual = ":weight_preprocess_hdrs",
    visibility = ["//visibility:public"],
)

# Interface-only dependency bundle. Engine targets (KVCacheManager etc., linked into
# libth_transformer.so) depend on this; they must NOT pull the stateful _impl
# targets, because the `g_*_fn` callback trio and the CudaRuntime legacy globals
# live in exactly one .so (librtp_compute_ops.so via th_compute_lib -> :_impl).
# A second copy in libth_transformer.so would be unregistered -> crash at
# KVCacheManager::allocateAndSync -> execAllGather assert.
cc_library(
    name = "exec_ops_hdr",
    deps = [
        ":op_data",
        ":device_data",
        ":weight_preprocess",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models:sampling_ops",
        "//rtp_llm/cpp/runtime:bootstrap",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/comm:collective_backend_hdrs",
        "//rtp_llm/cpp/cache:cache_store_writer",
        "//rtp_llm/cpp/core:copy_ops",
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_util",
    ] + torch_deps() + select({
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers"],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Test aggregator; tests link the _impl targets directly because they produce
# a single cc_test binary and need the symbol definitions locally.
cc_library(
    name = "exec_ops_test_lib",
    deps = [
        ":exec_ops_hdr",
        "//rtp_llm/cpp/runtime:cuda_runtime_impl",
        "//rtp_llm/cpp/runtime:bootstrap_impl",
        "//rtp_llm/cpp/comm:collective_backend_impl",
        "//rtp_llm/cpp/cache:cache_store_writer_impl",
        "//rtp_llm/cpp/core:copy_ops_impl",
        ":weight_preprocess_impl",
        "//rtp_llm/cpp/models:sampling_ops_impl",
        "//rtp_llm/cpp/models:stats_impl",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
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
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
    ],
    visibility = ["//visibility:public"],
)

# exec_ctx_ops: platform sample kernels (CudaSampleOp.cc / CudaBeamSearchOp.cc).
# Linked into librtp_compute_ops.so via cuda_impl (CUDA) or directly via
# th_compute_lib / common (ROCm).  The legacy exec_ctx_state / exec_ctx_rocm
# targets, which existed only to keep process-wide globals on a single .so,
# are gone now that those globals live in cpp/runtime + cpp/comm.
cc_library(
    name = "exec_ctx_ops",
    srcs = [
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
            "//rtp_llm/models_py/bindings/common/kernels:kernels_copy",
            "//3rdparty/trt_beam_search:trt_beam_search_impl",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "rejection_sampling_ops_test_lib",
    srcs = [
        "CudaSampleOp.cc",
    ],
    deps = [
        ":common_defines",
        ":op_data",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
    ] + torch_deps() + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/rocm:speculative_sampling",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
            "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
            "//3rdparty/flashinfer:flashinfer",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "cache_store_async_writer",
    hdrs = ["CacheStoreAsyncWriter.h"],
    srcs = ["CacheStoreAsyncWriter.cc"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:device_pin",
        "@havenask//aios/autil:lock_free",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
