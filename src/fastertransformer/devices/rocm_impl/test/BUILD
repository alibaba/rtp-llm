load("//:def.bzl", "copts", "torch_deps")
load("//src/fastertransformer/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
]

test_deps = [
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
