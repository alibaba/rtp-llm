package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package")
exports_files(["requirements_base.txt"])
py_package(
    name = "extension_package",
    deps = [
    ],
    packages = [
        "maga_transformer"
    ],
)

py_library(
    name = "extension_lib",
    deps = [],
)
