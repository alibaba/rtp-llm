package(default_visibility = ["//visibility:public"])

# Python deps (triton, torch, etc.) are managed by pip/pyproject.toml.
# This BUILD only declares package boundaries for Bazel-visible test targets.

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
)

# Aiter and FlyDSL are installed by the Python-native ROCm wheel dependencies.
# Keep the source files in the Bazel package so AMD tests and packaging see the
# same module inventory as upstream main.
py_library(
    name = "fla",
    srcs = glob(["fla/*.py"]),
    deps = [":common"],
    visibility = ["//visibility:public"],
)

# Include the patched source in the RTP wheel, not just Bazel's external tree.
genrule(
    name = "aiter_gdr_decode_padding_source",
    srcs = ["@aiter//:gdr_decode_source"],
    outs = ["fla/_aiter_gdr_decode_padding.py"],
    cmd = "cp $(location @aiter//:gdr_decode_source) $@",
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
    visibility = ["//visibility:public"],
)
