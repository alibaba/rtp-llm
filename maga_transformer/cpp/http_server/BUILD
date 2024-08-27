cc_library(
    name = "http_server",
    srcs = glob(["http_server/*.cpp"]),
    hdrs = glob(["http_server/*.h"]),
    include_prefix = "http_server",
    strip_include_prefix = "http_server",
    visibility = ["//visibility:public"],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:lock_free",
        "@havenask//aios/autil:json",
        "@havenask//aios/network/anet",
    ],
)
