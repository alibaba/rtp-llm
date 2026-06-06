load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@xgrammar//:xgrammar",
]

cc_test(
    name = "xgrammar_sanity_test",
    srcs = ["XGrammarSanityTest.cc"],
    deps = test_deps,
    copts = copts(),
)

cc_test(
    name = "xgrammar_backend_cpp_test",
    srcs = ["XGrammarBackendCppTest.cc"],
    deps = test_deps + [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
    ],
    copts = copts() + ["-fno-access-control"],
)
