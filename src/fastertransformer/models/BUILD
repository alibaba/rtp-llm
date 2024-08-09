load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
]

cc_library(
    name = "weights_define",
    hdrs = [
        "W.h",
    ],
    visibility = ["//visibility:public"],
)
