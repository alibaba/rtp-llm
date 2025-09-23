load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()


test_deps = [
    "//rtp_llm/cpp/devices/cpu_impl:cpu_impl",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/base_tests:base_tests",
] + torch_deps()

cc_test(
    name = "cpu_ffn_op_test",
    srcs = [
        "ops/CpuFFnOpTest.cc",
    ],
    data = [],
    tags = ["cpu", ],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_attention_layer_test",
    srcs = [
        "ops/CpuAttentionLayerTest.cc",
    ],
    data = [],
    tags = ["cpu", ],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_layernorm_test",
    srcs = [
        "ops/CpuLayerNormTest.cc",
    ],
    data = [],
    tags = ["cpu", ],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_embedding_test",
    srcs = [
        "ops/CpuEmbeddingTest.cc",
    ],
    data = [],
    tags = ["cpu", ],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_gemm_test",
    srcs = [
        "ops/CpuGemmTest.cc",
    ],
    data = [],
    tags = ["cpu", "manual"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_sampler_test",
    srcs = [
        "ops/CpuSamplerTest.cc",
    ],
    data = [],
    tags = ["cpu", "manual"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "cpu_ops_test",
    srcs = [
        "CpuOpsTest.cc",
    ],
    data = [],
    tags = ["cpu", ],
    copts = test_copts,
    deps = test_deps,
)
