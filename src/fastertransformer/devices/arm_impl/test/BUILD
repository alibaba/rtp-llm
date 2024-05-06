load("//:def.bzl", "copts", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
]

test_envs = {
    "TEST_USING_DEVICE": "ARM",
}

test_deps = [
    "//src/fastertransformer/devices/arm_impl:arm_cpu_impl",
    "//src/fastertransformer/devices/testing:device_test_utils",
] + torch_deps()

cc_test(
    name = "arm_cpu_basic_tests",
    env = test_envs,
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        "//src/fastertransformer/devices/base_tests:basic_test_cases"
    ],
)
