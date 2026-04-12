load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cuda_graph_base",
    hdrs = [
        "cuda_graph_base.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_srcs",
    srcs = [
        "cuda_graph_device_shims.cc",
        "cuda_graph_utils.cc",
        "cuda_graph_runner.cc",
        "cuda_graph_prefill.cc",
        "cuda_graph_decode.cc",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_hdrs",
    srcs = [
        "cuda_graph_base.h",
        "cuda_graph_device_shims.h",
        "cuda_graph_utils.h",
        "cuda_graph_runner.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_graph_hdrs_lib",
    hdrs = [
        "cuda_graph_base.h",
        "cuda_graph_device_shims.h",
        "cuda_graph_utils.h",
        "cuda_graph_runner.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_graph_impl",
    hdrs = [
        "cuda_graph_base.h",
        "cuda_graph_device_shims.h",
        "cuda_graph_utils.h",
        "cuda_graph_runner.h",
    ],
    srcs = [
        "cuda_graph_device_shims.cc",
        "cuda_graph_utils.cc",
        "cuda_graph_runner.cc",
        "cuda_graph_prefill.cc",
        "cuda_graph_decode.cc",
    ],
    deps = torch_deps() + [
        ":cuda_graph_base",
        ":cuda_graph_hdrs_lib",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/models_py/bindings:op_defs",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/cpp/cuda:cuda",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
