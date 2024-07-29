load("//:def.bzl", "rpm_library", "copts", "cuda_copts",)
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//visibility:public"])


cc_library(
    name = "cuda_driver",
    srcs = glob([
        "common/*.cpp",
    ]),
    hdrs = glob([
        "common/*.h",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//:__subpackages__"],
)

cc_library(
    name = "cub",
    hdrs = glob([
        "cub/**/*.h",
        "cub/**/*.cuh",
    ]),
    srcs = glob([
        "cub/**/*.cc",
        "cub/**/*.cu",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//src:__subpackages__"],
)
