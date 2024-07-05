load("//:def.bzl", "rocm_copts",)

cc_library(
    name = "rocm_types_hdr",
    hdrs = [
        "cuda_shims.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_utils",
    srcs = glob([
        "*.cc"
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
        "@local_config_rocm//rocm:rocblas",
        "//src/fastertransformer/utils:logger",
        "//src/fastertransformer/utils:assert_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:types",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
