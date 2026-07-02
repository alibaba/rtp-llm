package(default_visibility = ["//visibility:public"])

# Python deps (triton, torch, etc.) are managed by pip/pyproject.toml.
# This BUILD only declares package boundaries for Bazel-visible test targets.

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
)

# flydsl_*.py import the ROCm/MI308X-only `flydsl` package, which is not a
# generally available pip dependency. They are lazy-imported at runtime (see
# fla/chunk.py) and shipped via setup.py, so they are excluded from the Bazel
# srcs glob to avoid import-based dep resolution pulling in a nonexistent
# `@pip_gpu_rocm_torch//flydsl` package and breaking analysis.
py_library(
    name = "fla",
    srcs = glob(["fla/*.py"], exclude = ["fla/flydsl_*.py"]),
    deps = [":common"],
)

py_library(
    name = "kimi_kda",
    srcs = glob(["kimi_kda/*.py"]),
    deps = [
        ":common",
        ":fla",
        "//rtp_llm/models_py/triton_kernels/autotune_cache:autotune_cache",
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
