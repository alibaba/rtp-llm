cc_library(
    name = "common_service",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
        "@havenask//aios/autil:time",
    ],
    visibility = ["//visibility:public"]
)
