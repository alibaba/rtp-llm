package(default_visibility = ["//visibility:public"])

# Python deps (triton, torch, etc.) are managed by pip/pyproject.toml.
# This BUILD only declares package boundaries for Bazel-visible test targets.

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
)

py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"],
)

py_library(
    name = "kimi_kda",
    srcs = glob(["kimi_kda/*.py"]),
    deps = [
        ":common",
        ":fla",
    ],
)

py_library(
    name = "moe",
    srcs = glob(["moe/*.py"]),
    deps = [":common"],
)

py_library(
    name = "causal_conv1d",
    srcs = glob(["causal_conv1d/*.py"]),
)

py_library(
    name = "sparse_mla",
    srcs = glob(["sparse_mla/*.py"]),
)

py_library(
    name = "triton_kernels",
    deps = [
        ":common",
        ":fla",
        ":kimi_kda",
        ":moe",
        ":causal_conv1d",
        ":sparse_mla",
    ],
)
