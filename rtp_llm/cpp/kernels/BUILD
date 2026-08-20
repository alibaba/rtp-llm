load("//:def.bzl", "any_cuda_copts")

package(default_visibility = ["//visibility:public"])

device_deps = select({
    "@//:using_cuda": [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "//rtp_llm/cpp/runtime:rocm_compat",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "copy_kernels",
    srcs = [
        "BatchCopyKernel.cu",
        "MultiCopyKernel.cu",
    ],
    hdrs = [
        "BatchCopyKernel.h",
        "MultiCopyKernel.h",
    ],
    copts = any_cuda_copts(),
    deps = device_deps + [
        "//rtp_llm/cpp/runtime:device_error",
        "//rtp_llm/cpp/runtime:device_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "mask_logits_kernel",
    srcs = ["MaskLogitsKernel.cu"],
    hdrs = ["MaskLogitsKernel.h"],
    copts = any_cuda_copts(),
    deps = device_deps + [
        "//rtp_llm/cpp/runtime:device_error",
        "//rtp_llm/cpp/runtime:device_types",
    ],
)

cc_library(
    name = "fused_copy_kernel",
    srcs = ["FusedCopyKernel.cu"],
    hdrs = ["FusedCopyKernel.h"],
    copts = any_cuda_copts(),
    deps = device_deps + [
        "//rtp_llm/cpp/core:fused_copy_types",
        "//rtp_llm/cpp/runtime:device_types",
    ],
)

cc_library(
    name = "split_kv_copy_kernel",
    srcs = select({
        "@//:using_cuda": ["SplitKvCopyKernel.cu"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@//:using_cuda": ["SplitKvCopyKernel.h"],
        "//conditions:default": [],
    }),
    copts = any_cuda_copts(),
    deps = select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)
