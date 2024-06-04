load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "cutlass_kernels_interface")

cutlass_kernels_interface()

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
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/kernels:kernels",
        "cutlass_kernels_interface",
        "//3rdparty/flash_attention2:flash_attention2_header",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_header",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_header",
        "@havenask//aios/autil:string_helper",
    ],
    visibility = ["//visibility:public"],
    copts = cuda_copts(),
)
