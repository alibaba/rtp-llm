load("@//:def.bzl", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "deep_ep_hdrs",
    hdrs = glob([
        "csrc/*.hpp",
        "csrc/kernels/*.cuh",
    ]),
    includes = ["csrc"],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
        "@nvshmem//:nvshmem_hdrs",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)