load("//:def.bzl", "copts")

cc_library(
    name = "metrics",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)