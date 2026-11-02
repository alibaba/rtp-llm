load("//:def.bzl", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//rtp_llm:__subpackages__"])

cc_library(
    name = "sampling",
    srcs = ["api.cu", "sampler_dispatch.cc"],
    hdrs = [
        "sampling.h",
        "sampler_dispatch.h",
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
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
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
