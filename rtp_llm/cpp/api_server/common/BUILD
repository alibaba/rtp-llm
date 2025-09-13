cc_library(
    name = "common_service",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
        "@havenask//aios/autil:time",
    ],
    visibility = ["//visibility:public"]
)
