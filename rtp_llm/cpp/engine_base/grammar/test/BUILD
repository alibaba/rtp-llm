load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "xgrammar_backend_cpp_test",
    srcs = ["XGrammarBackendCppTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xgrammar//:xgrammar",
    ],
)

cc_test(
    name = "xgrammar_string_length_guard_test",
    srcs = ["XGrammarStringLengthGuardTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xgrammar//:xgrammar",
    ],
)

cc_test(
    name = "xgrammar_grammar_sanitize_test",
    srcs = ["XGrammarGrammarSanitizeTest.cc"],
    # Drives the backend's private static sanitize helpers directly: they are pure string rewrites, so
    # testing them through a real compile would only add a tokenizer and a compiler to the fixture.
    copts = copts() + ["-fno-access-control"],
    deps = [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
