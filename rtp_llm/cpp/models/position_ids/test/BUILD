load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()


cc_test(
    name = "position_ids_generator_test",
    srcs = glob([
        "PositionIdsGeneratorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/config:config_modules",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
