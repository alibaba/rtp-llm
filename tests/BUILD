load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//bazel:arch_select.bzl", "cutlass_kernels_interface")

cutlass_kernels_interface()
################################ cpp test ################################

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

cc_library(
    name = "test_lib",
    deps = torch_deps() + [
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/devices:devices_base_impl",
        "cutlass_kernels_interface",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "@flash_attention//:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
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
        "cutlass_kernels_interface",
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
    ],
)

py_test(
    name = "rotary_position_embedding",
    srcs = [
        "rotary_embedding/rotary_position_embedding.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:einops"
    ],
    tags = ["exclusive"]
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
    ],
)
