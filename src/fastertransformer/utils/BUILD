load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

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
    deps = [],
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
        ":assert_utils",
        "//src/fastertransformer/core:buffer"
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

