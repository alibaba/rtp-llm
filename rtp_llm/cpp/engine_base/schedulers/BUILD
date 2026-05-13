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
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:py_obj_utils",
    ],
    visibility = ["//visibility:public"],
)
