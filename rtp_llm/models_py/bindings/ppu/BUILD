load("//:def.bzl", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "ppu_silu_mul_mxfp4_op",
    srcs = ["PpuSiluMulMxfp4Op.cc"],
    hdrs = ["PpuSiluMulMxfp4Op.h"],
    copts = cuda_copts(),
    deps = [
        "//rtp_llm/models_py/bindings/common:common",
        "//rtp_llm/models_py/bindings/ppu/kernels:ppu_silu_mul_mxfp4",
    ] + torch_deps(),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "no_block_copy",
    srcs = ["NoBlockCopy.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)


cc_library(
    name = "ppu_bindings_register",
    srcs = [
        "RegisterPpuOps.cc",
        "//rtp_llm/models_py/bindings/core:exec_ops_srcs",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        "//rtp_llm/models_py/bindings/cuda:cuda_bindings_base",
        "//rtp_llm/models_py/bindings/cuda:cuda_attn_bindings_base",
        "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_ep_util",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/cpp/distribute:cpu_tp_broadcaster",
        ":no_block_copy",
        ":ppu_silu_mul_mxfp4_op",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
