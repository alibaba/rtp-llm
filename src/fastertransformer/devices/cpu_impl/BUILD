load("//:def.bzl", "copts")

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
        "//3rdparty/xft:xft_header",
        "//3rdparty/xft:libxfastertransformer",
    ],
    visibility = ["//visibility:public"],
    copts = copts() + ["-mavx512f", "-mavx512bw", "-mavx512vl"],
)
