load("//:def.bzl", _rpm_library = "rpm_library")

# Re-export for BUILD files that `load("//bazel:defs.bzl", "rpm_library")`.
rpm_library = _rpm_library

def torch_deps():
    return [
        "@torch//:torch_api",
        "@torch//:torch",
        "@torch//:torch_libs",
    ]
