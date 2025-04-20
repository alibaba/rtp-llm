load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")


cc_library(
    name = "stats",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/core:buffer"
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)