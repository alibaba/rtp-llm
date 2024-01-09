load("//:def.bzl", "copts", "cuda_copts", "torch_deps")
 
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
    hdrs = ["unittests/unittest_utils.h", "unittests/gtest_utils.h"],
    deps = torch_deps() + [
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/utils:torch_utils",
        "//src/fastertransformer/utils:nccl_utils",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "//3rdparty/flash_attention2:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "@com_google_googletest//:gtest",
    ],
    alwayslink = True,
)

cc_test(
    name = "test_activation",
    srcs = ["unittests/test_activation.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "test_attention_kernels",
    srcs = ["unittests/test_attention_kernels.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
    tags = ["exclusive"]
)

cc_test(
    name = "test_context_decoder_layer",
    srcs = ["unittests/test_context_decoder_layer.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "test_gemm",
    srcs = ["unittests/test_gemm.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "test_gpt_kernels",
    srcs = ["unittests/test_gpt_kernels.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "test_int8",
    srcs = ["unittests/test_int8.cu", "unittests/test_logprob_kernels.cu", 
            "unittests/test_penalty_kernels.cu",
            "unittests/test_tensor.cu"],
    deps = [":test_lib"],
    copts = test_copts,
    linkopts = test_linkopts,
    visibility = ["//visibility:public"],
)

# TODO(xinfei.sxf) fix those case, assert failed
# cc_test(
#     name = "test_sample",
#     srcs = ["unittests/test_sampling.cu", "unittests/test_sampling_layer.cu", 
#             "unittests/test_sampling_kernels.cu"],
#     deps = [":test_lib"],
#     copts = test_copts,
#     linkopts = test_linkopts,
#     visibility = ["//visibility:public"],
# )

################################ py test ################################

cc_library(
    name = "test_ops_libs",
    srcs = glob([
        "layernorm/*.cpp",
        "rotary_embedding/*.cpp",
        "attention_logn/*.cpp",
        "gemm_dequantize/*.cc",
        "gemm_group/*.cc",
        "unittests/test_activation.cu",
        "moe/*.cc"
    ]),
    hdrs = glob([
        "unittests/unittest_utils.h",
    ]),
    deps = torch_deps() + [
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils:torch_utils",
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

py_test(
    name = "test_th_decode_op",
    srcs = [
        "unittests/th_op/test_th_decode_op.py"
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
    name = "th_gemm_group",
    srcs = [
        "gemm_group/th_gemm_group.py"
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
    name = "th_moe",
    srcs = [
        "moe/th_moe.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    timeout = "long"
)