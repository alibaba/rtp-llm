load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_copts = copts()

sampler_test_copts = [
    # SamplerTest validates private buffer state; keep access override scoped to this target.
    "-fno-access-control",
] + test_copts


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
    copts = sampler_test_copts,
    deps = test_deps + [
        ":test_utils",
    ],
    env = device_test_envs(),
    # Keeps upstream's select shape and its ROCm branch; the CUDA branch is A10
    # rather than H20. sampler_test is in the 183 targets executed twice on sm86
    # and is not among that run's failures, and ut-sm8x has gone green with it
    # on A10. Upstream was preserving the old H20 pin while refactoring env to
    # device_test_envs(), not asserting this needs sm90.
    exec_properties = select({
        "@//:using_rocm": {"gpu": "MI308X-ROCM7"},
        "//conditions:default": {"gpu": "A10"},
    }),
)
