load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp:normal_engine",
    "//rtp_llm/cpp:dataclass",
    "//rtp_llm/cpp:stream",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
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
