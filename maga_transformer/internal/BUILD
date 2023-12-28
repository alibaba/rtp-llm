load("@rules_python//experimental/python:wheel.bzl", "py_package", "py_wheel")
load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

load("@pip_gpu_torch//:requirements.bzl", requirement_gpu_torch="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12_torch="requirement")
load("//bazel:defs.bzl", "upload_pkg", "copy_target_to", "upload_wheel", "pyc_wheel")

package(default_visibility = ["//maga_transformer:__subpackages__"])

py_library(
    name = "internal_lib",    
    srcs = glob([
        "__init__.py"
    ]),
    deps = [
    ],
)



