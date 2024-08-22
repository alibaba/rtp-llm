load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
]

test_deps = [
    "//src/fastertransformer/devices/cpu_impl:cpu_impl",
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/base_tests:base_tests",
] + torch_deps()

cc_test(
    name = "cpu_ffn_op_test",
    srcs = [
        "ops/CpuFFnOpTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cpu_attention_layer_test",
    srcs = [
        "ops/CpuAttentionLayerTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cpu_layernorm_test",
    srcs = [
        "ops/CpuLayerNormTest.cc",
    ],
    data = [],
    tags = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cpu_embedding_test",
    srcs = [
        "ops/CpuEmbeddingTest.cc",
    ],
    data = [],
    tags = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cpu_gemm_test",
    srcs = [
        "ops/CpuGemmTest.cc",
    ],
    data = [],
    tags = ["manual"],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)

cc_test(
    name = "cpu_ops_test",
    srcs = [
        "CpuOpsTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)
