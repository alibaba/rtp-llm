load("//:def.bzl", "cuda_copts")

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
