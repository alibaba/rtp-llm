load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

test_deps = [
    "//src/fastertransformer/devices:device_factory",
    "//src/fastertransformer/devices/testing:device_test_utils",
] + torch_deps()

cc_test(
    name = "cuda_ops_test",
    srcs = [
        "CudaOpsTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
)
