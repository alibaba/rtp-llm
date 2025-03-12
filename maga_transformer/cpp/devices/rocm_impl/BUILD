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
        "@aiter//:decode_mla",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "@local_config_rocm//rocm:rccl",
        "@composable_kernel//:ck_fmha_example",
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp/devices:devices_base_impl",
        "//maga_transformer/cpp/core:allocator",
        "//maga_transformer/cpp/rocm:rocm_types_hdr",
        "//maga_transformer/cpp/rocm:rocm_utils",
        "//maga_transformer/cpp/kernels:hello_kernel",
        "//maga_transformer/cpp/kernels:gpt_kernels_rocm",
        "//maga_transformer/cpp/cuda:nccl_util",
        "//:th_utils",
        "//maga_transformer/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    copts = rocm_copts(),
    alwayslink = True,
)
