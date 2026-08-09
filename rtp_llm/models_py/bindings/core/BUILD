load("//:def.bzl", "rpm_library", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "types_hdr",
    hdrs = [
        "Types.h",
        "Dispatch.h",
    ],
    deps = [
        "//rtp_llm/cpp/core:memory_types",
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
    deps = ["//rtp_llm/cpp/core:op_status"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_holder",
    hdrs = [
        "TensorHolder.h",
    ],
    deps = torch_deps(),
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
        "//rtp_llm/cpp/cache:cache_store_types",
        "//rtp_llm/cpp/comm:collective_types",
        "//rtp_llm/cpp/core:copy_types",
        "//rtp_llm/cpp/models:sampling_types",
        ":type_convert",
        ":common_defines",
        ":tensor_holder",
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

# Stateless compatibility implementation. Keep it in librtp_compute_ops.so.
cc_library(
    name = "weight_preprocess_impl",
    srcs = ["WeightPreprocess.cc"],
    deps = [":weight_preprocess_hdrs"] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//rtp_llm/cpp/pybind:__pkg__"],
)

cc_library(
    name = "runtime_ops_test_impls",
    deps = [
        "//rtp_llm/cpp/cache:cache_store_writer_impl",
        "//rtp_llm/cpp/comm:collective_backend_impl",
        "//rtp_llm/cpp/core:copy_ops_impl",
        "//rtp_llm/cpp/models:stats_impl",
        "//rtp_llm/cpp/runtime:cuda_runtime_impl",
        ":weight_preprocess_impl",
    ],
    copts = copts(),
    visibility = [
        "//rtp_llm/cpp/cache/connector/memory/test:__pkg__",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test:__pkg__",
        "//rtp_llm/cpp/cache/connector/remote_connector/test:__pkg__",
        "//rtp_llm/cpp/cache/connector/test:__pkg__",
        "//rtp_llm/cpp/cache/test:__pkg__",
        "//rtp_llm/cpp/disaggregate/cache_store/test:__pkg__",
        "//rtp_llm/cpp/models/logits_processor/test:__pkg__",
        "//rtp_llm/cpp/models/test:__pkg__",
        "//rtp_llm/cpp/testing:__pkg__",
        "//rtp_llm/models_py/bindings/core/test:__pkg__",
        "//rtp_llm/models_py/bindings/rocm/ops/tests:__pkg__",
    ],
)

cc_library(
    name = "sampling_ops_test_impls",
    deps = [
        "//rtp_llm/cpp/models:sampling_ops_impl",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "@//:using_rocm": [
            ":sampling_ops_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = [
        "//rtp_llm/cpp/api_server/test:__pkg__",
        "//rtp_llm/cpp/cache/connector/p2p/test:__pkg__",
        "//rtp_llm/cpp/disaggregate/cache_store/test:__pkg__",
        "//rtp_llm/cpp/engine_base/schedulers/test:__pkg__",
        "//rtp_llm/cpp/engine_base/stream/test:__pkg__",
        "//rtp_llm/cpp/engine_base/system_prompt/test:__pkg__",
        "//rtp_llm/cpp/model_rpc/test:__pkg__",
        "//rtp_llm/cpp/models/logits_processor/test:__pkg__",
        "//rtp_llm/cpp/models/position_ids/test:__pkg__",
        "//rtp_llm/cpp/models/test:__pkg__",
        "//rtp_llm/cpp/multimodal_processor/test:__pkg__",
        "//rtp_llm/cpp/normal_engine/speculative/test:__pkg__",
        "//rtp_llm/cpp/normal_engine/test:__pkg__",
        "//rtp_llm/cpp/test:__pkg__",
        "//rtp_llm/models_py/bindings/core/test:__pkg__",
        "//rtp_llm/models_py/bindings/cuda/ops/tests:__pkg__",
        "//rtp_llm/models_py/bindings/rocm/ops/tests:__pkg__",
    ],
)

# DeviceData.h - standalone header for DeviceInitParams, DeviceType, etc.
cc_library(
    name = "device_data",
    hdrs = [
        "DeviceData.h",
    ],
    deps = [
        ":common_defines",
        "//rtp_llm/cpp/runtime:device_status",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:config_modules",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)


filegroup(
    name = "sampling_ops_impl_srcs",
    srcs = [
        "CudaSampleOp.cc",
        "CudaBeamSearchOp.cc",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling_ops_impl",
    srcs = [":sampling_ops_impl_srcs"],
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
        "//rtp_llm/cpp/models:sampling_ops",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
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
        "//rtp_llm/cpp/models:sampling_ops",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
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
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_store_writer_hdrs",
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:device_pin",
        "//rtp_llm/models_py/bindings:op_defs",
        "@havenask//aios/autil:lock_free",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)
