load("@//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda",
    srcs = ["csrc/fast_hadamard_transform_cuda.cu"],
    hdrs = glob(["csrc/*.h"]),
    includes = ["csrc"],
    copts = copts() + cuda_default_copts_without_arch() + [
        "-O3",
        "--cuda-gpu-arch=sm_120",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "-nvcc_options=expt-relaxed-constexpr",
        "-nvcc_options=expt-extended-lambda",
        "-nvcc_options=use_fast_math",
    ],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "fast_hadamard_transform",
    srcs = ["csrc/fast_hadamard_transform.cpp"],
    copts = copts() + ["-DRTP_LLM_EMBEDDED_FHT"],
    deps = [":cuda"] + torch_deps(),
)
