load("//:def.bzl", "copts")

cc_library(
    name = "schedulers",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base/stream",
        "//rtp_llm/cpp/engine_base/system_prompt",
    ],
)
