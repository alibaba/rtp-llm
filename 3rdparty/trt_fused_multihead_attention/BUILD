load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "trt_fused_multihead_attention_impl",
    srcs = glob([
        "*.cpp",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "@cutlass//:cutlass",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//3rdparty:cuda_driver",
    ],
    copts = copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)

cc_library(
    name = "trt_fused_multihead_attention_header",
    hdrs = glob([
        "qkvToContext.h",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
