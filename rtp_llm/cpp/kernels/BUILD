load("//:def.bzl", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//rtp_llm:__subpackages__"])

cuda_deps = [
    "//rtp_llm/cpp/cuda:cuda_host_utils",
    ":cuda_utils_common",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
]

rocm_deps = [
    ":rocm_utils",
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:hip",
    "//rtp_llm/cpp/rocm:rocm_types_hdr",
    "//rtp_llm/cpp/rocm:rocm_host_utils",
]

any_cuda_deps = select({
    "@//:using_cuda": cuda_deps,
    "@//:using_rocm": rocm_deps,
    "//conditions:default": [],
})

# =============================================================================
# CUDA common utilities
# =============================================================================

cc_library(
    name = "cuda_utils_common",
    hdrs = [
        "vec_dtypes.cuh",
        "util.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

# =============================================================================
# CUDA-only kernels - split by functionality
# =============================================================================

cc_library(
    name = "cuda_fused_qk_rmsnorm",
    srcs = ["fused_qk_rmsnorm.cu"],
    hdrs = ["fused_qk_rmsnorm.h"],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
    ],
    copts = any_cuda_copts(), 
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_no_aux_tc",
    srcs = [
        "no_aux_tc_kernels.cu",
    ],
    hdrs = [
        "no_aux_tc_kernels.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:launch_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = cuda_copts(),
)

cc_library(
    name = "scaled_fp8_quant",
    srcs = [
        "scaled_fp8_quant.cu",
    ],
    hdrs = [
        "scaled_fp8_quant.h",
        "scaled_fp8_quant_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_quantization",
    srcs = [
        "per_token_group_quant_8bit.cu",
    ],
    hdrs = [
        "per_token_group_quant_8bit.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
)

cc_library(
    name = "cuda_moe",
    srcs = [
        "moe_topk_softmax_kernels.cu",
    ],
    hdrs = [
        "moe_topk_softmax_kernels.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
)

cc_library(
    name = "cuda_comm",
    srcs = [
        "comm_buffer.cu",
    ],
    hdrs = [
        "comm_buffer.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = cuda_copts(),
)

cc_library(
    name = "cuda_cu",
    deps = [
        ":cuda_no_aux_tc",
        ":cuda_quantization",
        ":cuda_moe",
        ":cuda_comm",
        ":cuda_fused_qk_rmsnorm",
    ],
    visibility = ["//visibility:public"],
)

# =============================================================================
# ROCm kernels - split by functionality
# =============================================================================

cc_library(
    name = "rocm_utils",
    hdrs = glob([
        "rocm_utils/*.h",
    ]),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

# ROCm LayerNorm kernels
cc_library(
    name = "rocm_layernorm",
    srcs = [
        "rocm/layernorm_kernels.cu",
    ],
    hdrs = [
        "rocm/layernorm_kernels.h",
    ],
    deps = rocm_deps + [
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)

# ROCm quantization kernels
cc_library(
    name = "rocm_quantization",
    srcs = [
        "rocm/quantization_rocm.cu",
    ],
    hdrs = [
        "rocm/quantization_rocm.h",
    ],
    deps = rocm_deps + [
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)

# ROCm fused QK RMSNorm kernels
cc_library(
    name = "rocm_fused_qk_rmsnorm",
    srcs = [
        "rocm/fused_qk_rmsnorm.cu",
    ],
    hdrs = [
        "rocm/fused_qk_rmsnorm.h",
    ],
    deps = rocm_deps + [
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)

# Aggregated ROCm basic kernels
cc_library(
    name = "rocm_basic",
    deps = [
        ":rocm_layernorm",
        ":rocm_quantization",
        ":rocm_fused_qk_rmsnorm",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_mla",
    srcs = glob([
        "mla_kernels_rocm/*.cu",
    ]),
    hdrs = glob([
        "mla_kernels_rocm/*.h",
    ]),
    deps = rocm_deps,
    copts = rocm_copts(),
)

cc_library(
    name = "rocm_cu",
    deps = [
        ":rocm_basic",
        ":rocm_mla",
    ],
    visibility = ["//visibility:public"],
)

# =============================================================================
# Common kernels - split by functionality
# =============================================================================

cc_library(
    name = "kernels_activation",
    srcs = [
        "activation_kernels.cu",
    ],
    hdrs = [
        "activation_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_layernorm",
    srcs = [
        "layernorm_kernels.cu",
        "rmsnormKernels.cu",
        #"fused_qk_rmsnorm.cu",
    ],
    hdrs = [
        "layernorm_kernels.h",
        "rmsnormKernels.h",
        "fused_qk_rmsnorm.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
    ] + select({
        "//:enable_triton": ["//rtp_llm/cpp/kernels/triton:triton_kernel"],
        "//conditions:default": [],
     }),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_sampling",
    srcs = [
        "sampling_topk_kernels.cu",
        "sampling_topp_kernels.cu",
        "sampling_penalty_kernels.cu",
        "ban_bad_words.cu",
        "banRepeatNgram.cu",
        "mask_logits.cu",
    ],
    hdrs = [
        "sampling_topk_kernels.h",
        "sampling_topp_kernels.h",
        "sampling_penalty_kernels.h",
        "ban_bad_words.h",
        "banRepeatNgram.h",
        "mask_logits.h",
        "penalty_types.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
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
        "kv_cache/kv_cache_index.h"
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
    name = "kernels_attention",
    srcs = [
        "unfused_attention_kernels.cu",
    ],
    hdrs = [
        "unfused_attention_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
        ":attn_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

# =============================================================================
# Refactored modular kernels (from original gpt_kernels)
# =============================================================================

cc_library(
    name = "kernels_embedding",
    srcs = [
        "embedding_kernels.cu",
    ],
    hdrs = [
        "embedding_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_tensor_ops",
    srcs = [
        "tensor_ops_kernels.cu",
    ],
    hdrs = [
        "tensor_ops_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_kv_cache",
    srcs = [
        "kv_cache_kernels.cu",
    ],
    hdrs = [
        "kv_cache_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_residual",
    srcs = [
        "add_residual_kernels.cu",
    ],
    hdrs = [
        "add_residual_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:launch_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
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
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_custom_ar",
    srcs = [
        "custom_ar_kernels.cu",
    ],
    hdrs = [
        "custom_ar_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_quantization",
    srcs = [
        "quantization_tensor.cu",
    ],
    hdrs = [
        "quantization_tensor.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_moe",
    srcs = [
        "moe_kernels.cu",
    ],
    hdrs = [
        "moe_kernels.h",
    ],
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_ep_util",
    srcs = glob([
        "moe/*.cu",
    ]),
    hdrs = glob([
        "moe/*.h",
    ]),
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/cuda/cutlass:cutlass_interface",
        "//rtp_llm/cpp/cuda:trt_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@cutlass_h_moe//:cutlass",
        "@cutlass_h_moe//:cutlass_utils",
        "//rtp_llm/cpp/cuda/cutlass/cutlass_extensions:cutlass_extensions",
    ]+ torch_deps(),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_eplb",
    srcs = glob([
        "eplb/*.cu",
    ]),
    hdrs = glob([
        "eplb/*.h",
    ]),
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_mla",
    srcs = glob([
        "mla_kernels/*.cu",
    ]),
    hdrs = glob([
        "mla_kernels/*.h",
    ]),
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
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
        ":kernels_attention",
        ":kernels_embedding",
        ":kernels_tensor_ops",
        ":kernels_kv_cache",
        ":kernels_residual",
        ":kernels_copy",
        ":kernels_custom_ar",
        ":kernels_quantization",
        ":kernels_moe",
        ":kernels_eplb",
        ":kernels_mla",
        "//rtp_llm/cpp/kernels/decoder_masked_multihead_attention:mmha",
    ] + select({
        "@//:using_cuda": [
            ":cuda_cu",
        ],
        "@//:using_rocm": [
            ":rocm_cu",
        ],
        "//conditions:default": [],
    }) + any_cuda_deps + torch_deps(),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
