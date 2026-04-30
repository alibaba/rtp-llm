load("//:def.bzl", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "custom_allreduce",
    srcs = ["custom_allreduce.cu"],
    hdrs = [
        "custom_allreduce.h",
        "custom_allreduce.cuh",
    ],
    copts = cuda_copts(),
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda",
    ],
    visibility = ["//visibility:public"],
)
