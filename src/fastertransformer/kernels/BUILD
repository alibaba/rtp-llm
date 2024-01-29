load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "kernels_cu",
    srcs = glob([
        "*.cu",
        "decoder_masked_multihead_attention/*.cu",
        "online_softmax_beamsearch/*.cu",
    ], exclude = [
        "layernorm_fp8_kernels.cu",
        "activation_fp8_kernels.cu",
        "unfused_attention_fp8_kernels.cu",
        "moe_kernels.cu",
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
        "decoder_masked_multihead_attention/*.h",
        "online_softmax_beamsearch/*.h",
        "online_softmax_beamsearch/*.hpp"
    ], exclude=[
        "AttentionFP8Weight.h",
        "moe_kernels.h",
    ]),
    deps = [
        "//src/fastertransformer/cuda:memory_utils",
        "//src/fastertransformer/cuda:nvtx",
        "//src/fastertransformer/cuda:cuda_utils",
        "//src/fastertransformer/utils:utils",
        "//3rdparty:cub",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels",
    srcs = glob([
        "*.cc",
        "decoder_masked_multihead_attention/*.cc",
    ], exclude=[
        "**/*_test.cc",
    ]),
    deps = [
        ":kernels_cu",
        "//3rdparty:cuda_driver",
        "//src/fastertransformer/utils:utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

