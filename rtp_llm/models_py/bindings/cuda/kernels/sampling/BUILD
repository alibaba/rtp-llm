load("//:def.bzl", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//rtp_llm:__subpackages__"])

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
        ":split_softmax",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

# PyTorch softmax preserves FP32 subnormals. Override the engine's default FTZ
# only for this new compilation unit; existing FlashInfer sampling stays intact.
cc_library(
    name = "split_softmax",
    srcs = ["split_softmax.cu"],
    hdrs = ["split_softmax.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts() + ["-nvcc_options=ftz=false"],
    visibility = ["//visibility:public"],
)
