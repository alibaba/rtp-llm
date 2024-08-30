load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")


cc_library(
    name = "test_lib",
    deps = torch_deps() + [
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/models:models",
        "@flash_attention//:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/contextFusedMultiHeadAttentionSm70:trt_sm70_fmha_impl",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

################################ py test ################################

cc_library(
    name = "test_ops_libs",
    srcs = glob([
        "layernorm/*.cpp",
        "rotary_embedding/*.cpp",
        "attention_logn/*.cpp",
        "gemm_dequantize/*.cc",
        "gemm_group/*.cc",
        "int8_gemm/*.cc",
    ]),
    deps = torch_deps() + [
        "//src/fastertransformer/kernels:kernels",
	"//:th_utils",
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "test_ops",
    deps = [":test_ops_libs"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "generalT5LayerNorm",
    srcs = [
        "layernorm/generalT5LayerNorm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:transformers",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "rotary_position_embedding",
    srcs = [
        "rotary_embedding/rotary_position_embedding.py",
        "rotary_embedding/yarn_rotary_embedding.py",
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:transformers",
        "//maga_transformer:einops"
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "logn_attention",
    srcs = [
        "attention_logn/logn_attention.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "th_int8_gemm",
    srcs = [
        "int8_gemm/th_int8_gemm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops"
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:numpy",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "th_gemm_dequantize",
    srcs = [
        "gemm_dequantize/th_gemm_dequantize.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:numpy",
    ],
    exec_properties = {'gpu':'A10'},
)
