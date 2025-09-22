load("//bazel:arch_select.bzl", "requirement")

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
    ]),
    deps = [
        "//rtp_llm:utils",
    ] + select({
        "@//:using_cuda12":["@deep_gemm_ext//:deep_gemm"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "distributed",
    srcs = glob([
        "distributed/*.py",
    ]),
    deps = [
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:process_group_state",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ]),
    deps = [
        ":utils",
        ":kernels",
        ":distributed",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules_cuda",
    srcs = glob([
        "modules/cuda/*.py",
    ]),
)

py_library(
    name = "modules_rocm",
    srcs = glob([
        "modules/rocm/*.py",
    ]),
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py"
    ]),
    deps = [
        ":utils",
        ":modules",
        "//rtp_llm/model_loader:loader",
    ] + select({
        "@//:using_cuda12": [
            ":modules_cuda",
            "@deep_gemm_ext//:deep_gemm"
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
