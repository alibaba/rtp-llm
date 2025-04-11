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
        "//src/fastertransformer/devices/testing:device_test_utils",
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
        "//src/fastertransformer/devices/testing:device_test_utils",
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
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/torch_impl:torch_reference_impl",
        "//maga_transformer/cpp:models",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)
