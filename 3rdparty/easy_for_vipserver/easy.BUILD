cc_import(
    name = "libeasy",
    static_library = "deps/lib/libeasy.a",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libeasy_headers",
    hdrs = glob([
        "deps/include/easy/*.h",
    ]),
    strip_include_prefix = "deps/include/",
    visibility = ["//visibility:public"],
)
