load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "grammar_cpp_lib",
    hdrs = [
        "RtpGrammarMatcher.h",
        "XGrammarBackend.h",
        "XGrammarBootstrap.h",
    ],
    srcs = [
        "RtpGrammarMatcher.cc",
        "XGrammarBackend.cc",
        "XGrammarBootstrap.cc",
    ],
    deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:special_tokens",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
)

