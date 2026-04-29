load("//:def.bzl", "rpm_library")

def torch_deps():
    return [
        "@torch//:torch_api",
        "@torch//:torch",
        "@torch//:torch_libs",
    ]
