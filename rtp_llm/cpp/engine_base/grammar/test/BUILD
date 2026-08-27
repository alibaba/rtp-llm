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
