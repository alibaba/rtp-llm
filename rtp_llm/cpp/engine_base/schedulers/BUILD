load("//:def.bzl", "copts")

cc_library(
    name = "schedulers",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/cache:cache",
    ],
    visibility = ["//visibility:public"],
)
