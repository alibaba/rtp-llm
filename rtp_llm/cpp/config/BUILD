load("//:def.bzl", "copts")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "static_config",
    hdrs = ["StaticConfig.h"],
    srcs = ["StaticConfig.cc"],
    copts = copts(),
)

cc_library(
    name = "config_modules",
    hdrs = ["ConfigModules.h"],
    srcs = ["ConfigModules.cc"],
    deps = [
        "@havenask//aios/autil:env_util",
    ],
    copts = copts(),
)

cc_library(
    name = "eplb_config",
    hdrs = ["EplbConfig.h"],
    copts = copts(),
)
