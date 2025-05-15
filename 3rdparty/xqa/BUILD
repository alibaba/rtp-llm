load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")

cc_library(
    name = "xqa_impl",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_driver",
    ],
    copts = cuda_copts() + ['--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a', "--expt-relaxed-constexpr"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xqa_header",
    hdrs = glob([
        "mha.h",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
