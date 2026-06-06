load("//:def.bzl", "copts")

# Native C++ grammar backend — port of the Python xgrammar/reasoner backends.
# Goal: let GrammarManager (and Phase-3 GenerateStream) reach grammar
# functionality without crossing the Python boundary in any hot path.
#
# Dependency direction:
#   * grammar_cpp depends ONLY on @xgrammar (the third-party library) and
#     RTP-LLM's logging utilities. It does NOT depend on schedulers/, stream/,
#     or any pybind11 layer — keeps it Python-free and reusable.
#   * GrammarManager (in the schedulers/ package) and any future call sites
#     consume grammar_cpp via :grammar_cpp_lib.
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
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
        "@xgrammar//:xgrammar",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
