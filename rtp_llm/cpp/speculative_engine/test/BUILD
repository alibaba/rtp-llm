load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lnccl",
    "-lnvToolsExt",
] + select({
    "@//:using_cuda": [
        "-ltorch_cuda",
        "-lc10_cuda",
    ],
    "@//:using_rocm": [
        "-ltorch_hip",
        "-lc10_hip",
    ],
    "//conditions:default": [
    ],
})

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
        "//rtp_llm/cpp:speculative_engine",
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
    linkopts = test_linkopts,
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
    linkopts = test_linkopts,
    deps =  [
        ":sp_mock_engine",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
