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
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"]
)
