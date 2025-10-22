load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "select_py_bindings", "torch_deps")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "py_utils",
    srcs = [
        "PyUtils.cc",
    ],
    hdrs = [
        "PyUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

cc_library(
    name = "th_transformer_config_lib",
    srcs = [
        "ConfigInit.cc",
        "common/blockUtil.cc",
    ],
    hdrs = [
        "common/blockUtil.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":py_utils",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:hash_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "th_compute_lib",
    srcs = [
        "ComputeInit.cc",
    ],
    hdrs = [],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + torch_deps() + select_py_bindings(),
    alwayslink = True,
)

cc_library(
    name = "th_transformer_lib",
    srcs = [
        "common/InitEngineOps.cc",
        "init.cc",
        "multi_gpu_gpt/RtpEmbeddingOp.cc",
        "multi_gpu_gpt/RtpLLMOp.cc",
    ] + select({
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
        "common/InitEngineOps.h",
        "common/NcclOp.h",
        "common/blockUtil.h",
        "multi_gpu_gpt/RtpEmbeddingOp.h",
        "multi_gpu_gpt/RtpLLMOp.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/api_server:http_api_server",
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
    ] + torch_deps(),
    alwayslink = True,
)

cc_library(
    name = "th_transformer_gpu",
    srcs = select({
        "@//:using_cuda12": [
            "common/CutlassConfigOps.cc",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cuda/cutlass:cutlass_kernels_common",
    ] + torch_deps(),
    alwayslink = True,
)

cc_library(
    name = "th_utils",
    hdrs = [
        "th_utils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
)
