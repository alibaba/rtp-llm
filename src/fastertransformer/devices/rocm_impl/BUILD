load("//:def.bzl", "rocm_copts")

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
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/rocm:rocm_types_hdr",
        "//src/fastertransformer/rocm:rocm_utils",
        "//src/fastertransformer/kernels:hello_kernel",
        "//src/fastertransformer/kernels:gpt_kernels_rocm",
        "//src/fastertransformer/cuda:nccl",
        "//:th_utils",
    ],
    visibility = ["//visibility:public"],
    copts = rocm_copts(),
    alwayslink = True,
)
