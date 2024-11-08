load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "utils",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = torch_deps() + [
        "//src/fastertransformer/utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)


