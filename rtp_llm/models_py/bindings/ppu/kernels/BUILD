load("//:def.bzl", "cuda_copts")

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
