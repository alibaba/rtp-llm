load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()


cc_library(
    name = "sp_mock_engine",
    hdrs = [
        "SpeculativeMockEngine.h"
    ],
    srcs = [
    ],
    copts = test_copts,
    deps =  [
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "//rtp_llm/cpp/speculative_engine:speculative_engine",
    ] + torch_deps(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "sp_engine_test",
    srcs = glob([
        "SpeculativeEngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps =  [
        ":sp_mock_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "sp_sampler_test",
    srcs = glob([
        "SpeculativeSamplerTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps =  [
        ":sp_mock_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
