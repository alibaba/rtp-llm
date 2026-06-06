load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")

alias(
    name = "autotune_cache",
    actual = "//rtp_llm/models_py/triton_kernels/autotune_cache",
    visibility = ["//visibility:public"],
)

py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"] + select({
        "@rtp_llm//:using_rocm": [requirement_gpu_rocm("flydsl")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"]
)

py_library(
    name = "kimi_kda",
    srcs = glob(["kimi_kda/*.py"]),
    deps = [
        ":autotune_cache",
        ":common",
        ":fla",
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "causal_conv1d",
    srcs = glob(["causal_conv1d/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "moe",
    srcs = glob(["moe/*.py"]),
    deps = ["//rtp_llm/models_py:utils"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "sparse_mla",
    srcs = glob(["sparse_mla/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "grammar",
    srcs = glob(["grammar/*.py"]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "triton_kernels",
    deps = [
        ":autotune_cache",
        ":fla",
        ":kimi_kda",
        ":causal_conv1d",
        ":moe",
        ":common",
        ":sparse_mla",
        ":grammar",
    ],
    visibility = ["//visibility:public"]
)