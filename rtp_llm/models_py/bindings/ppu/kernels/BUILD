load("//:def.bzl", "copts", "cuda_copts")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ppu_rope_kvcache",
    srcs = ["fused_rope_kvcache_kernel.cu"],
    hdrs = ["fused_rope_kvcache_kernel.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/common/kernels:attn_utils",
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_utils_common",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
