load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# Cooks xgrammar's TokenizerInfo opaque blob; private to this package.
cc_library(
    name = "grammar_xgrammar_cooker",
    hdrs = ["XGrammarTokenizerInfoCooker.h"],
    srcs = ["XGrammarTokenizerInfoCooker.cc"],
    deps = [
        "@havenask//aios/autil:json",
        "@xgrammar//:xgrammar",
    ],
    visibility = ["//visibility:private"],
    copts = copts(),
)

# Public grammar API: zero xgrammar symbols in exported headers.
cc_library(
    name = "grammar_api",
    hdrs = [
        "TokenizerInfo.h",
    ],
    srcs = [
        "TokenizerInfo.cc",
    ],
    deps = [
        ":grammar_xgrammar_cooker",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:special_tokens",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = copts(),
)

# xgrammar-specific impl; only Tier 0 (this package) and Tier 1 (logits_processor) may depend.
cc_library(
    name = "grammar_xgrammar",
    hdrs = [
        "RtpGrammarMatcher.h",
        "XGrammarBackend.h",
    ],
    srcs = [
        "RtpGrammarMatcher.cc",
        "XGrammarBackend.cc",
    ],
    deps = [
        ":grammar_api",
        ":grammar_xgrammar_cooker",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:special_tokens",
        "//rtp_llm/cpp/utils:core_utils",
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
)

# Transitional umbrella; deleted once callers move onto :grammar_api or :grammar_xgrammar.
cc_library(
    name = "grammar_cpp_lib",
    deps = [
        ":grammar_api",
        ":grammar_xgrammar",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
