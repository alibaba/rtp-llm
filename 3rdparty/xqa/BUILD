load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
        "//maga_transformer/cpp/cuda:cuda_utils",
        "//3rdparty:cuda_driver",
    ],
    copts = cuda_copts() + ['--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a', "--expt-relaxed-constexpr"],
    visibility = ["//:__subpackages__"],
    alwayslink = True,
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
