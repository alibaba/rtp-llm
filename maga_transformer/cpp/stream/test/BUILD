load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//maga_transformer/cpp:normal_engine",
    "//maga_transformer/cpp:dataclass",
    "//maga_transformer/cpp:stream",
    "//maga_transformer/cpp/devices/testing:device_test_utils",
    "//maga_transformer/cpp/devices/cuda_impl:cuda_impl",
    "//:gpt_init_params",
] + torch_deps()

cc_test(
    name = "stream_cache_resource_test",
    srcs = [
        "StreamCacheResourceTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
