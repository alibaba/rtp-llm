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
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    linkopts = [
        "-lpython3.10",
        "-ltorch",
        "-lc10",
    ],
)

cc_library(
    name = "torch_beam_search_op_impl",
    hdrs = glob([
        "BeamSearchOp.h"
    ]),
    deps = torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
    linkopts = [
        "-lpython3.10",
        "-ltorch",
        "-lc10",
    ],
)
