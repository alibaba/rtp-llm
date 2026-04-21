load("//:def.bzl", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//rtp_llm:__subpackages__"])

any_cuda_deps = select({
    "@//:using_cuda": [
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_utils_common",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "//rtp_llm/models_py/bindings/rocm/kernels:rocm_utils",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
        "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "kernels_activation",
    srcs = ["activation_kernels.cu"],
    hdrs = ["activation_kernels.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_sampling",
    srcs = [
        "sampling_penalty_kernels.cu",
        "banRepeatNgram.cu",
        "mask_logits.cu",
    ],
    hdrs = [
        "sampling_penalty_kernels.h",
        "banRepeatNgram.h",
        "mask_logits.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "attn_utils",
    hdrs = [
        "decoder_masked_multihead_attention_utils.h",
        "rotary_position_embedding.h",
        "kv_cache/kv_cache_utils.h",
        "kv_cache/kv_cache_index.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_layernorm",
    srcs = ["layernorm_kernels.cu"],
    hdrs = [
        "layernorm_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_debug",
    srcs = ["debug_kernel.cu"],
    hdrs = ["debug_kernel.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        ":attn_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_embedding",
    srcs = ["embedding_kernels.cu"],
    hdrs = ["embedding_kernels.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_moe",
    srcs = ["moe_kernels.cu"],
    hdrs = ["moe_kernels.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_tensor_ops",
    srcs = ["tensor_ops_kernels.cu"],
    hdrs = ["tensor_ops_kernels.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_kv_cache",
    srcs = ["kv_cache_kernels.cu"],
    hdrs = ["kv_cache_kernels.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_copy",
    srcs = [
        "batch_copy.cu",
        "copy_utils.cu",
        "cuda_graph_copy_kernel.cu",
    ],
    hdrs = [
        "batch_copy.h",
        "copy_utils.h",
        "cuda_graph_copy_kernel.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_ep_util",
    srcs = glob(["moe/*.cu"]),
    hdrs = glob(["moe/*.h"]),
    deps = any_cuda_deps + [
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
    ] + torch_deps(),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sm_copy_kernel",
    srcs = ["sm_copy_kernel.cu"],
    hdrs = ["sm_copy_kernel.h"],
    copts = any_cuda_copts(),
    deps = select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fuse_copy_util",
    hdrs = [
        "fuse_copy_util.h",
    ],
    deps = any_cuda_deps,
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fuse_copy_kernel",
    srcs = [
        "fuse_copy_kernel.cu",
    ],
    hdrs = [
        "fuse_copy_kernel.h",
    ],
    deps = any_cuda_deps + [
        ":fuse_copy_util",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_cu",
    deps = [
        ":kernels_activation",
        ":kernels_layernorm",
        ":kernels_sampling",
        ":kernels_debug",
        ":kernels_embedding",
        ":kernels_tensor_ops",
        ":kernels_kv_cache",
        ":kernels_copy",
        ":kernels_moe",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/kernels:cuda_cu",
            "//rtp_llm/models_py/bindings/cuda/kernels:kernels_mla",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm/kernels:rocm_cu",
        ],
        "//conditions:default": [],
    }) + any_cuda_deps + torch_deps(),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
