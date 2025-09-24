load("//:def.bzl", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "rocm_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "@local_config_rocm//rocm:rccl",
        "@composable_kernel//:ck_fmha_example",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:devices_base_impl",
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
        "//rtp_llm/cpp/rocm:rocm_host_utils",
        "//rtp_llm/cpp/rocm:rocm_custom_ar",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "//rtp_llm/cpp/cuda:nccl_util",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ] + torch_deps() + select({
        "@//:using_aiter_src": [
            "@aiter_src//:module_aiter_enum",
            "@aiter_src//:module_quant",
            "@aiter_src//:module_gemm_a8w8_blockscale",
            "@aiter_src//:module_gemm_a8w8_bpreshuffle",
            "@aiter_src//:module_moe_sorting",
            "@aiter_src//:module_moe_asm",
            "@aiter_src//:module_pa",
            "@aiter_src//:module_moe",
            "@aiter_src//:module_activation",
        ],
        "//conditions:default": [
            "@aiter//:module_aiter_enum",
            "@aiter//:module_quant",
            "@aiter//:module_gemm_a8w8_blockscale",
            "@aiter//:module_gemm_a8w8_bpreshuffle",
            "@aiter//:module_moe_sorting",
            "@aiter//:module_moe_asm",
            "@aiter//:module_pa",
            "@aiter//:module_moe",
            "@aiter//:module_activation",
        ],
    }),
    visibility = ["//visibility:public"],
    copts = rocm_copts(),
    alwayslink = True,
)
