load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()


cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/engine_base:weights_converter",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + device_impl_target()

cc_library(
    name = "test_utils",
    hdrs = [
        "ModelTestUtil.h",
    ],
    srcs = [
        "ModelTestUtil.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "py_testdata",
    srcs = glob([
        "python/*",
        "python/**/*",
    ]),
)

cc_test(
    name = "sampler_test",
    srcs = [
        "SamplerTest.cc",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata"
    ],
    copts = test_copts,
    deps = test_deps + [
        ":test_utils",
    ],
    env = select({
        "@//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
    exec_properties = {'gpu':'H20'},
)
