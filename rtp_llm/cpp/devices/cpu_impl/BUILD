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
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/core:cpu_allocator",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/devices:devices_base_impl",
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
