load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xgrammar_tokenizer_info",
    hdrs = [
        "XGrammarTokenizerInfo.h",
    ],
    srcs = [
        "XGrammarTokenizerInfo.cc",
    ],
    deps = [
        "@havenask//aios/autil:json",
        "@xgrammar//:xgrammar",
    ],
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
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:special_tokens",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
)

# Transitional umbrella; deleted once callers move onto :grammar_xgrammar.
cc_library(
    name = "grammar_cpp_lib",
    deps = [
        ":grammar_xgrammar",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
