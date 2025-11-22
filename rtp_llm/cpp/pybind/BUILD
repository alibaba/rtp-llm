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
    ],
    hdrs = [],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + torch_deps() + select_py_bindings(),
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
    ] +  select({
        "@//:using_cuda12": [
            "common/NcclOp.cc",
        ],
        "@//:using_cuda": [
            "common/NcclOp.cc",
        ],
        "@//:using_rocm": [
            "common/NcclOp.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "common/NcclOp.h",
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
        "//:rtp_compute_ops",
    ] + torch_deps(),
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