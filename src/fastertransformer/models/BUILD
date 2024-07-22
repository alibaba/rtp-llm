load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
]

cc_library(
    name = "weights_define",
    hdrs = [
        "W.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "models",
    srcs = glob([
        "multi_gpu_gpt/*.cc",
    ], exclude = [ # test and fp8 codes
        "multi_gpu_gpt/gpt_gemm.cc",
    ]),
    hdrs = glob([
        "multi_gpu_gpt/*.h",
    ]),
    deps = [
        ":weights_define",
        "//:gpt_init_params_hdr",
        "@flash_attention//:flash_attention2_header",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_header",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_header",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/cuda:expert_attention_util",
        "//src/fastertransformer/cuda:cuda",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
