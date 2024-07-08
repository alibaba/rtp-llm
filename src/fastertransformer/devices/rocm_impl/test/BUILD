load("//:def.bzl", "copts", "torch_deps", "rocm_copts")
load("//src/fastertransformer/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + rocm_copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-L/opt/rocm/lib",
    "-lhipblas",
    "-lhipblaslt",
]

test_deps = [
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:rocblas",
    "//src/fastertransformer/devices/rocm_impl:rocm_impl",
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/base_tests:base_tests",
] + torch_deps()

cc_test(
    name = "rocm_basic_test",
    srcs = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/devices/base_tests:basic_test_cases"
    ],
    tags = ["rocm"],
)

cc_test(
    name = "rocm_ops_test",
    srcs = [
        "RocmOpsTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
)

cc_test(
    name = "gemm_op_test",
    srcs = [
        "ops/ROCmGemmOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
)

cc_test(
    name = "rocm_act_op_test",
    srcs = [
        "ops/RocmActOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
)

cc_test(
    name = "rocm_ffn_op_test",
    srcs = [
        "ops/RocmFFnOpTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
)

cc_test(
    name="rocm_attention_op_test",
    srcs=[
        "ops/ROCmAttentionOpTest.cc",
    ],
    data=[],
    env=device_test_envs(),
    copts=test_copts + copts(),
    linkopts=test_linkopts,
    deps=test_deps,
)

cc_test(
    name = "embedding_lookup_test",
    srcs = [
        "ops/EmbeddingLookupTest.cc",
    ],
    data = [],
    env = device_test_envs(),
    copts = test_copts + copts(),
    linkopts = test_linkopts,
    deps = test_deps,
    tags = ["rocm"],
)
