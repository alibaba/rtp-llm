load("//:def.bzl", "copts", "cuda_copts", "torch_deps")


test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

test_envs = {
    "TEST_USING_DEVICE": "CUDA",
}

cc_library(
    name = "cuda_test_util",
    hdrs = [
        "CudaTestUtils.h",
    ],
    srcs = [],
    deps = [
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
    copts = cuda_copts() + copts(),
)

test_deps = [
    "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
    "//3rdparty/flash_attention2:flash_attention2_impl",
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/base_tests:base_tests",
    ":cuda_test_util",
] + torch_deps()


cc_test(
    name = "cuda_basic_test",
    srcs = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/devices/base_tests:basic_test_cases"
    ],
)

cc_test(
    name = "cuda_dist_test",
    srcs = [],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "FT_DISABLE_CUSTOM_AR": "1"
    },
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/devices/base_tests:distributed_test"
    ],
    tags = ["multi_device"],
)

cc_test(
    name = "cuda_custom_all_reduce_test",
    srcs = [
        "CudaCustomAllReduceTest.cc"
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["multi_device"],
)

cc_test(
    name = "cuda_general_ops_test",
    srcs = [
        "CudaGeneralOpsTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_gemm_op_test",
    srcs = [
        "CudaGemmOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_group_gemm_op_test",
    srcs = [
        "ops/CudaGroupGemmOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_act_op_test",
    srcs = [
        "CudaActOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_ffn_op_test",
    srcs = [
        "CudaFFnOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_attention_op_test",
    srcs = [
        "CudaAttentionOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_attention_layer_test",
    srcs = [
        "CudaAttentionLayerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_softmax_op_test",
    srcs = [
        "CudaSoftmaxOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "layernorm_test",
    srcs = [
        "LayernormTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps,
)