load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "torch_reference_impl",
    hdrs = glob([
        "*.hpp", "*.h"
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "torch_beam_search_op_impl",
    hdrs = glob([
        "BeamSearchOp.h"
    ]),
    deps = torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)
