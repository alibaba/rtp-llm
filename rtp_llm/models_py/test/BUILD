package(default_visibility = ["//visibility:public"])

exports_files(["defs.bzl", "test_main.py"])

py_library(
    name = "test_runner",
    srcs = ["__init__.py", "test_runner.py"],
)

py_test(
    name = "kernel_layout_test",
    srcs = ["kernel_layout_test.py"],
    data = [
        "//rtp_llm/models_py:kernel_layout_sources",
    ],
)
