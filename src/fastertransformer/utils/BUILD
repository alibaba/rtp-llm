load("//:def.bzl", "rpm_library", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "logger",
    srcs = [
        "logger.cc",
        "exception.cc"
    ],
    hdrs = [
        "logger.h",
        "string_utils.h",
        "exception.h"
    ],
    deps = ["@havenask//aios/alog:alog"],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "assert_utils",
    hdrs = [
        "assert_utils.h"
    ],
    deps = [
        ":logger"
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = torch_deps() + [
        ":logger",
        ":assert_utils"
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "py_utils",
    srcs = glob([
        "py_utils/*.cc",
    ]),
    hdrs = glob([
        "py_utils/*.h",
    ]),
    deps = torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
