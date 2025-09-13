load("//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("@//3rdparty/xqa:def.bzl", "compile_xqa_libs")

genrule(
    name = "gen_header",
    cmd = "/opt/conda310/bin/python $< --output $@",
    outs = ["mha_sm90.h"],
    srcs = ["gen_header.py"],
    tags = ["local"],
)

xqa_libs = compile_xqa_libs()

cc_library(
    name = "xqa",
    srcs = ["mha_dispatch_sm90.cu"],
    hdrs = glob([
        "*.h",
        "*.cuh",
    ]) + [":gen_header"],
    deps = xqa_libs,
    copts = cuda_copts() + [
        '--cuda-include-ptx=sm_90a',
        '--cuda-gpu-arch=sm_90a',
        "-nvcc_options=expt-relaxed-constexpr",
        '-nvcc_options=use_fast_math',
        '-t 0',
        '-res-usage',
    ],
    visibility = ["//visibility:public"],
)
