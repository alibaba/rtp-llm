cc_library(
    name = "xft_pip",
    srcs = [
         "xfastertransformer-devel/lib/libxfastertransformer.so",
    ],
    hdrs = glob([
        "xfastertransformer-devel/include/*.h",
    ]),
    deps = [
        "@local_config_python//:python_headers",
    ],
    strip_include_prefix = "xfastertransformer-devel",
    include_prefix = "xfastertransformer",
    visibility = ["//visibility:public"],
)
