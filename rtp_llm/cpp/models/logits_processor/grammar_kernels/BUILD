load("//:def.bzl", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xgrammar_kernels",
    srcs = ["xgrammar_kernels.cu"],
    hdrs = ["xgrammar_kernels.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
)
