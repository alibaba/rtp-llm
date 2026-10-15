load("//:def.bzl", "copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")

cc_library(
    name = "w4a16_sm120",
    srcs = ["gemm_api.cu", "weight_transform.cu"],
    hdrs = glob(["**/*.cuh"]),
    copts = copts() + cuda_default_copts_without_arch() + [
        "-O3",
        "--cuda-gpu-arch=sm_120a",
        "-nvcc_options=use_fast_math",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)
