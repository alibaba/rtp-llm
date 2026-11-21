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
        "//rtp_llm/cpp/models:context_parallel_layout",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:stage0_gap_audit",
    ],
    visibility = ["//visibility:public"],
)
