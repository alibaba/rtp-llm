load("//:def.bzl", "any_cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "sm_copy_kernel",
    srcs = ["sm_copy_kernel.cu"],
    hdrs = ["sm_copy_kernel.h"],
    copts = any_cuda_copts(),
    deps = select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    visibility = ["//visibility:public"],
)
