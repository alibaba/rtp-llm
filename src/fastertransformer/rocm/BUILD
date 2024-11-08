load("//:def.bzl", "rocm_copts",)

cc_library(
    name = "rocm_types_hdr",
    hdrs = [
        "cuda_shims.h",
        "amd_bfloat16.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_utils",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
    ]),
    deps = [
        "@local_config_rocm//rocm:rocm",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:types",
        "@composable_kernel//:ck_fmha_example",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "rocm",
    deps = [
        "//src/fastertransformer/cuda:nvtx",
        ":rocm_utils",
    ],
    copts=rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name="memory_utils",
    srcs=[
        # "memory_utils.cu",
    ],

    hdrs=[
        # "memory_utils.h",
    ],

    deps=[
        ":rocm_utils",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/core:allocator",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
    ],
    copts=rocm_copts(),
    include_prefix="src",
    visibility=["//visibility:public"],
)