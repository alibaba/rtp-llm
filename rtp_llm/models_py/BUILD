load("//bazel:arch_select.bzl", "requirement", "deep_gemm_deps")

deep_gemm_deps()

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
    ])
)

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
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
            ":modules_cuda"
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12":["deep_gemm"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
