load("//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("@//3rdparty/xqa:def.bzl", "compile_xqa_libs")

xqa_libs = compile_xqa_libs()

cc_library(
    name = "xqa",
    srcs = glob([
        "mha_dispatch_sm90.cu",
    ]),
    hdrs = [
        "mha.h"
    ],
    deps = xqa_libs,
    copts = cuda_copts() + ['--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a', "--expt-relaxed-constexpr"],
    visibility = ["//visibility:public"],
)
