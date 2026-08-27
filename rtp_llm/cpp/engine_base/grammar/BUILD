load("//:def.bzl", "copts")

cc_library(
    name = "grammar_cpp_lib",
    hdrs = [
        "RtpGrammarMatcher.h",
        "XGrammarBackendCpp.h",
    ],
    srcs = [
        "RtpGrammarMatcher.cc",
        "XGrammarBackendCpp.cc",
    ],
    deps = [
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock_free",
        "@xgrammar//:xgrammar",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
