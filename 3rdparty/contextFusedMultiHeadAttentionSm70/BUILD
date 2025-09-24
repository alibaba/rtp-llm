load("//:def.bzl", "rpm_library", "copts", "cuda_copts",)
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trt_sm70_fmha_impl",
    srcs = glob([
        "fmhaRunner.cpp",
        "**/*.cpp"
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        ":trt_sm70_fmha_header",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_common_define_header",
        "//3rdparty:cuda_driver",
        "//rtp_llm/cpp/kernels:attn_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)


cc_library(
    name = "trt_sm70_fmha_header",
    hdrs = glob([
        "*.h",
        "**/*.h"
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
