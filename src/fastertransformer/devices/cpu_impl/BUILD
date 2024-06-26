load("//:def.bzl", "copts")

#TODO(wangyin): maybe add build option to select icx version of xft-devel package.

cc_library(
    name = "cpu_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/core:cpu_allocator",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/devices:devices_base_impl",
        "@xfastertransformer_devel_icx//:xft_pip",
    ],
    visibility = ["//visibility:public"],
    copts = copts() + ["-mavx512f", "-mavx512bw", "-mavx512vl"],
    #alwayslink = 1,
)
