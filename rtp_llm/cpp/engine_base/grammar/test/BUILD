load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "xgrammar_backend_cpp_test",
    srcs = ["XGrammarBackendCppTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xgrammar//:xgrammar",
    ],
    env = {
        "TEST_USING_DEVICE": "CPU",
    },
)
