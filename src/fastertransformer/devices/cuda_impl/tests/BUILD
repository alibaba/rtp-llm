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
    env = test_envs,
    copts = cuda_copts() + copts() + torch_deps(),
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/devices/base_tests:distributed_test"
    ],
    tags = ["multi_device"],
)

cc_test(
    name = "cuda_ops_test",
    srcs = [
        "CudaOpsTest.cc",
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
        "ops/CudaGemmOpTest.cc",
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
        "ops/CudaActOpTest.cc",
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
        "ops/CudaFFnOpTest.cc",
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
        "ops/CudaAttentionOpTest.cc",
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
        "ops/CudaSoftmaxOpTest.cc",
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
    name = "embedding_lookup_test",
    srcs = [
        "EmbeddingLookupTest.cc",
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


