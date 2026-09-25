load("//:def.bzl", "rocm_copts", "any_cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
package(default_visibility = ["//rtp_llm:__subpackages__"])

rocm_deps = [
    "//rtp_llm/models_py/bindings/rocm/kernels:rocm_utils",
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:hip",
    "//rtp_llm/models_py/bindings/dcu:dcu_types_hdr",
    "//rtp_llm/models_py/bindings/dcu:dcu_host_utils",
]

cc_library(
    name = "rocm_layernorm",
    srcs = ["layernorm_kernels.cu"],
    hdrs = ["layernorm_kernels.h"],
    deps = rocm_deps + [
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_fused_qk_rmsnorm",
    srcs = ["fused_qk_rmsnorm.cu"],
    hdrs = ["fused_qk_rmsnorm.h"],
    deps = rocm_deps + [
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_basic",
    deps = [
        ":rocm_layernorm",
        ":rocm_fused_qk_rmsnorm",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_cu",
    deps = [
        ":rocm_basic",
        "//rtp_llm/models_py/bindings/rocm/kernels/sampling:sampling",
    ],
    visibility = ["//visibility:public"],
)
