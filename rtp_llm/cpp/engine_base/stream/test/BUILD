load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/engine_base:engine_base",
    "//rtp_llm/cpp/engine_base/stream:stream",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
    "//rtp_llm/cpp/cache/connector/test/mock:connector_mock_files_lib",
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

cc_test(
    name = "generate_stream_test",
    srcs = [
        "GenerateStreamTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
