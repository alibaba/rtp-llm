load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps", "select_py_bindings")

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
        "@//:using_cuda12": ["//rtp_llm/cpp/core:exec_ops_srcs"],
        "//conditions:default": [],
    }),
    hdrs = [],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/core:devices_base",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:device_perf_wrapper",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + torch_deps() + select_py_bindings() + select({
        "@//:using_cuda12": ["//rtp_llm/cpp/cuda/ops:gpu_base"],
        "@//:using_rocm": ["//rtp_llm/cpp/core:exec_ctx_rocm"],
        "//conditions:default": [],
    }),
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
        "@//:using_cuda12": ["//rtp_llm/cpp/cuda/ops:gpu_base"],
        # libth_transformer.so does NOT contain its own copy of
        # globalRuntime() / globalPgState().  Those symbols are
        # resolved at load-time from librtp_compute_ops.so.
        "@//:using_rocm": [
            "//rtp_llm/cpp/core:exec_ctx_ops",
            "//rtp_llm/cpp/core:exec_ops_hdr",
            "//rtp_llm/cpp/core:distributed_comm_hdr",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_gpu",
    srcs = select({
        "@//:using_cuda12": [
            "common/CutlassConfigOps.cc",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/cuda/cutlass:cutlass_kernels_common",
    ] + torch_deps(),
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