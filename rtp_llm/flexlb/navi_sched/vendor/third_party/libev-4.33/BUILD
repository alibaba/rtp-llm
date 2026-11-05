cc_library(
    name = "libev_headers",
    hdrs = ["ev.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "libev",
    shared_library = "libev.so",
    deps = [":libev_headers"],
    visibility = ["//visibility:public"],
)
