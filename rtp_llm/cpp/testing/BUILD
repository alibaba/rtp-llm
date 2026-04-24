load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(
        ["*.cc"],
        exclude = ["BasicDeviceTest.cc", "DistributedTest.cc"],
    ),
    deps = [
        "@com_google_googletest//:gtest",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "basic_test_cases",
    srcs = ["BasicDeviceTest.cc"],
    deps = [
        ":device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "distributed_test",
    srcs = ["DistributedTest.cc"],
    deps = [
        ":device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "base_tests",
    hdrs = ["BeamSearchOpTest.hpp"],
    deps = [
        ":device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:torch_beam_search_op_impl",
        "//rtp_llm/cpp/models:models",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)
