load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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
        "//rtp_llm/cpp:position_ids_generator",
        "//rtp_llm/cpp:models",
        "//rtp_llm/cpp:stream",
        "//:gpt_init_params",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
