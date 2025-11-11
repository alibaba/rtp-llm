cc_library(
    name = "common_service",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/api_server:http_server",
    ],
    visibility = ["//visibility:public"]
)
