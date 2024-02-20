load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "cuda_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//3rdparty/flash_attention2:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    ],
    visibility = ["//visibility:public"],
    copts = cuda_copts(),
)
