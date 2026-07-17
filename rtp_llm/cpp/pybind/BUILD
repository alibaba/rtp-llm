load("//:def.bzl", "copts")
load(
    "@arch_config//:arch_select.bzl",
    "no_block_copy_link_deps",
    "select_py_bindings",
    "torch_deps",
)

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "py_utils",
    hdrs = [
        "PyUtils.h",
    ],
    srcs = [
        "PyUtils.cc",
    ],
    deps = [
        "//:rtp_compute_ops",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_config_lib",
    srcs = [
        "common/blockUtil.cc",
        "ConfigInit.cc",
    ],
    hdrs = [
        "common/blockUtil.h",
    ],
    deps = [
        ":py_utils",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/multimodal_processor:multimodal_input_class",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",  # Add ModelConfig implementation
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:hash_util",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_compute_lib",
    srcs = [
        "ComputeInit.cc",
    ] + select({
        "@//:using_cuda12": ["//rtp_llm/models_py/bindings/core:exec_ops_srcs"],
        "@//:using_xpu": ["//rtp_llm/models_py/bindings/core:exec_ops_srcs"],
        "//conditions:default": [],
    }),
    hdrs = [],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:device_perf_wrapper",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + torch_deps() + select_py_bindings() + select({
        "@//:using_cuda12": ["//rtp_llm/models_py/bindings/cuda/ops:gpu_base"],
        "@//:using_rocm": ["//rtp_llm/models_py/bindings/core:exec_ctx_rocm"],
        "@//:using_xpu": [
            "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
            "//rtp_llm/models_py/bindings/core:device_data",
            "//rtp_llm/models_py/bindings/core:op_data",
            "//rtp_llm/models_py/bindings/core:type_convert",
            "//rtp_llm/models_py/bindings/core:types_hdr",
            "//rtp_llm/models_py/bindings/core:common_defines",
            "//rtp_llm/cpp/utils:signal_utils",
            "//rtp_llm/cpp/utils:kv_cache_utils",
            "//rtp_llm/cpp/config:static_config",
            "//rtp_llm/cpp/model_utils:model_utils",
            "//rtp_llm/cpp/config:config_modules",
            "//rtp_llm/models_py/bindings:op_defs",
            "//rtp_llm/cpp/models:stats",
            "//rtp_llm/cpp/models/models_weight:weights_define",
            "//rtp_llm/cpp/cache:cache_group_type",
            "//rtp_llm/cpp/pybind:th_utils",
            "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
            "@havenask//aios/autil:stack_tracer",
            "@havenask//aios/autil:string_helper",
            "@havenask//aios/autil:scope",
        ],
        "//conditions:default": [],
    }) + no_block_copy_link_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_lib",
    srcs = [
        "init.cc",
        "common/InitEngineOps.cc",
        "multi_gpu_gpt/RtpEmbeddingOp.cc",
        "multi_gpu_gpt/RtpLLMOp.cc",
    ],
    hdrs = [
        "common/InitEngineOps.h",
        "common/blockUtil.h",
        "multi_gpu_gpt/RtpEmbeddingOp.h",
        "multi_gpu_gpt/RtpLLMOp.h",
    ],
    deps = [
        "//rtp_llm/cpp/api_server:http_api_server",
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/utils:signal_utils",
        "//:rtp_compute_ops",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        # libth_transformer.so does NOT contain its own copy of
        # globalRuntime().  Those symbols are resolved at load-time
        # from librtp_compute_ops.so.
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
            "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_xpu": [
            "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
            "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_gpu",
    srcs = [],
    deps = [] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_utils",
    hdrs = [
        "th_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)