py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"],
    visibility = ["//visibility:public"]
)

py_library(
    name = "kimi_kda",
    srcs = glob(["kimi_kda/*.py"]),
    deps = [":common"],
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
    name = "triton_kernels",
    deps = [
        ":fla",
        ":kimi_kda",
        ":causal_conv1d",
        ":moe",
        ":common",
        ":sparse_mla",
    ],
    visibility = ["//visibility:public"]
)