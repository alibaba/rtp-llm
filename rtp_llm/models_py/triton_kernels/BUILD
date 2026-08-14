py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"],
    visibility = ["//visibility:public"]
)

alias(
    name = "autotune_cache",
    actual = "//rtp_llm/models_py/triton_kernels/autotune_cache",
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
    visibility = ["//visibility:public"]
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
    name = "zigzag_permute",
    srcs = ["zigzag_permute.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "triton_kernels",
    deps = [
        ":fla",
        ":causal_conv1d",
        ":moe",
        ":common",
        ":sparse_mla",
        ":autotune_cache",
        ":kimi_kda",
        ":zigzag_permute",
    ],
    visibility = ["//visibility:public"]
)
