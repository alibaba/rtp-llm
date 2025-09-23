load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "basic_test_cases",
    srcs = glob([
        "BasicDeviceTest.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "distributed_test",
    srcs = glob([
        "DistributedTest.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "base_tests",
    hdrs = glob([
        "*.hpp",
    ]),
    deps = [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/torch_impl:torch_reference_impl",
        "//rtp_llm/cpp/models:models",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)
