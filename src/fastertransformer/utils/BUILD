load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

cc_library(
    name = "logger",
    srcs = [
        "logger.cc",
        "tllmException.cpp"
    ],
    hdrs = [
        "logger.h",
        "string_utils.h",
        "tllmException.h"
    ],
    deps = [],
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
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

