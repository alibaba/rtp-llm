load("//:def.bzl", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//rtp_llm:__subpackages__"])

cc_library(
    name = "dspark_logits",
    srcs = ["dspark_logits.cu"],
    hdrs = ["dspark_logits.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    # Match ATen's separately rounded FP32 add/divide, including subnormals.
    copts = cuda_copts() + [
        "-nvcc_options=ftz=false",
        "-nvcc_options=prec-div=true",
        "-nvcc_options=fmad=false",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling",
    srcs = ["api.cu"],
    hdrs = [
        "sampling.h",
        "flashinfer/air_top_p.cuh",
        "flashinfer/allocator.h",
        "flashinfer/exception.h",
        "flashinfer/math.cuh",
        "flashinfer/sampling.cuh",
        "flashinfer/topk.cuh",
        "flashinfer/topk_common.cuh",
        "flashinfer/utils.cuh",
        "flashinfer/vec_dtypes.cuh",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)
