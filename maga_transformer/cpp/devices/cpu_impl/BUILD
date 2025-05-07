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
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp/core:cpu_allocator",
        "//maga_transformer/cpp/devices:device_utils",
        "//maga_transformer/cpp/devices:devices_base_impl",
    ] + select({
        "//:xft_use_icx": [
            "@xfastertransformer_devel_icx//:xft_pip",
        ],
        "//conditions:default": [
            "@xfastertransformer_devel//:xft_pip",
        ],
    }),
    visibility = ["//visibility:public"],
    copts = copts() + ["-mavx512f", "-mavx512bw", "-mavx512vl"],
    alwayslink = 1,
)
