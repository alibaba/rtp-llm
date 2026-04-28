load("@//:def.bzl", "copts", "cuda_copts")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@//bazel:arch_select.bzl", "torch_deps")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")

flash_mla_cuda_copts = copts() + cuda_default_copts_without_arch() + if_cuda([
    "-D_USE_MATH_DEFINES", 
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-nvcc_options=expt-relaxed-constexpr",
    "-nvcc_options=expt-extended-lambda",
    # "-nvcc_options=ptxas-options=-v,--register-usage-level=10",
    "-nvcc_options=ptxas-options=--register-usage-level=10", # disable the verbose flag of ptxas for clear compile log
    "-nvcc_options=use_fast_math",
    '--cuda-include-ptx=sm_90', '--cuda-gpu-arch=sm_90',
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
])

flash_mla_headers = glob([
    "csrc/*.h",
])

cc_library(
    name = "flashmla_hdrs",
    hdrs = flash_mla_headers,
    deps = [
        "@cutlass//:cutlass",
        "@cutlass//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    includes = ["csrc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "flashmla_cu",
    srcs = flash_mla_headers + glob([
        'csrc/flash_fwd_mla_*.cu'
    ]),
    includes = ["csrc"],
    deps = [
        ':flashmla_hdrs'
    ] + torch_deps(),
    copts = flash_mla_cuda_copts,
)

cc_library(
    name = "flashmla_interface",
    hdrs = [
        "csrc/flashmla.h",
    ],
    strip_include_prefix = "csrc",
    include_prefix = "flashmla",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "flashmla",
    srcs = flash_mla_headers + glob([
        'csrc/flash_api.cpp'
    ]),
    includes = ["csrc"],
    deps = [
        ":flashmla_cu",
        ":flashmla_interface",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
