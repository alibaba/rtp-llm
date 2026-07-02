# The standalone Python package is built and tested through pip/setuptools.
# Python dependencies are managed in pyproject.toml; this BUILD file keeps only
# the Bazel-visible C++ runtime data and internal test helpers that still exist.

py_library(
    name = "auto_model",
    srcs = ["auto_model.py"],
    deps = [":py_standalone_testlib"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "py_standalone_testlib",
    deps = [
        "//rtp_llm/test/utils:test_util",
    ],
    data = [
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
