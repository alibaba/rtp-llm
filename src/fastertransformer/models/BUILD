load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

cc_library(
    name = "models",
    srcs = glob([
        "multi_gpu_gpt/*.cc",
    ], exclude = [ # test and fp8 codes
        "multi_gpu_gpt/gpt_gemm.cc",
    ]),
    hdrs = glob([
        "multi_gpu_gpt/*.h",
    ]) + [
        "W.h",
    ],
    deps = [
        "//:th_utils",
        "//3rdparty/flash_attention2:flash_attention2_header",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_header",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/cuda:cuda",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "gpt_gemm",
    srcs = [
        "multi_gpu_gpt/gpt_gemm.cc",
    ],
    deps = [
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/cuda:gemm_test_utils",
    ],
    copts = copts(),
    linkopts = [
        "-ldl"
    ],
    visibility = ["//visibility:public"],
)
