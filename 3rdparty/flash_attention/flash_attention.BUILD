load("@//:def.bzl", "cuda_copts", "copts")
load("@//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "flash_attention2_header",
    hdrs = [
        "@//3rdparty/flash_attention:flash_api.h",
        "csrc/flash_attn/src/flash.h",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "flash_attention2_impl",
    srcs = glob([
        "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
        "csrc/flash_attn/src/*.cpp"
    ]),
    hdrs = glob([
        "csrc/flash_attn/src/*.h",
        "csrc/flash_attn/src/*.cuh",
    ], exclude=["csrc/flash_attn/src/flash.h"]),
    deps = [
        "@cutlass_fa//:cutlass",
        ":flash_attention2_header",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "fa_hdrs",
    hdrs = glob([
        "csrc/flash_attn/src/*.h",
        "csrc/flash_attn/src/*.cuh",
    ], exclude=["csrc/flash_attn/src/flash.h"]),
    deps = torch_deps() + [
        "@cutlass_fa//:cutlass",
        ":flash_attention2_header",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_shared_library(
    name = "fa",
    roots = [":flash_attention2_impl"],
    preloaded_deps = torch_deps() + [
        "@cutlass_fa//:cutlass",
        ":flash_attention2_header",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)
