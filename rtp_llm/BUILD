package(default_visibility = ["//visibility:public"])

exports_files(["release_version.py"])

# ---------------------------------------------------------------------------
# Compatibility shims: Python packages are now managed by pip/pyproject.toml.
# These empty py_library targets satisfy Bazel's analysis phase for callers
# (e.g. rtp_llm/test/smoke/defs.bzl) that have not yet been migrated.
# ---------------------------------------------------------------------------

py_library(
    name = "uvicorn",
    srcs = [],
    deps = [],
)

py_library(
    name = "fastapi",
    srcs = [],
    deps = [],
)

py_library(
    name = "psutil",
    srcs = [],
    deps = [],
)

py_library(
    name = "tiktoken",
    srcs = [],
    deps = [],
)

py_library(
    name = "pydantic",
    srcs = [],
    deps = [],
)

py_library(
    name = "json5",
    srcs = [],
    deps = [],
)

py_library(
    name = "dashscope",
    srcs = [],
    deps = [],
)

py_library(
    name = "jieba",
    srcs = [],
    deps = [],
)

py_library(
    name = "partial_json_parser",
    srcs = [],
    deps = [],
)

py_library(
    name = "openai",
    srcs = [],
    deps = [],
)

py_library(
    name = "testlib",
    srcs = [],
    deps = [],
    data = [],
)

py_library(
    name = "sdk",
    srcs = [],
    deps = [],
    data = [],
)
