load("//:def.bzl", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "ppu_rope_kvcache",
    srcs = ["fused_rope_kvcache_kernel.cu"],
    hdrs = ["fused_rope_kvcache_kernel.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/common/kernels:attn_utils",
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_utils_common",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//rtp_llm/models_py/bindings/ppu:__pkg__"],
    alwayslink = 1,
)

# Apache-2.0 PPU AOT port from SGLang v0.5.14_release.
cc_library(
    name = "ppu_silu_mul_mxfp4",
    srcs = ["ppu_silu_mul_mxfp4.cu"],
    hdrs = ["ppu_silu_mul_mxfp4.h"],
    data = ["ppu_silu_mul_mxfp4.NOTICE"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//rtp_llm/models_py/bindings/ppu:__pkg__"],
)

# Qualified ordinary-MTP2 kernels. Keep their registered schemas and reduction
# orders identical to the measured native implementations.
cc_library(
    name = "mtp2_precision_kernels",
    srcs = [
        "compressor_cols4.cu",
        "fp32_row_projection.cu",
        "hc_prepare.cu",
        "hc_prepare_q3.cu",
        "hc_prepare_rms.cu",
        "hc_projection.cu",
        "hc_rms_tail.cu",
        "hc_row_reduce.cu",
    ],
    alwayslink = 1,
    copts = cuda_copts(),
    deps = [
        ":hc_rms_bf16_prefill",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//rtp_llm/models_py/bindings/ppu:__pkg__"],
)

# The measured HC XF extension used -O3; keep it separate from the other
# kernels' existing cuda_copts()/repository -O2 compilation recipe.
cc_library(
    name = "hc_rms_bf16_prefill",
    srcs = ["hc_rms_bf16_prefill.cu"],
    alwayslink = 1,
    copts = cuda_copts() + ["-O3"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:private"],
)
