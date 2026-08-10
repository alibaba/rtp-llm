load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@xgrammar//:xgrammar",
]

cc_test(
    name = "xgrammar_backend_cpp_test",
    srcs = ["XGrammarBackendTest.cc"],
    deps = test_deps + [
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "//rtp_llm/cpp/engine_base/grammar:xgrammar_tokenizer_info",
    ],
    copts = copts() + ["-fno-access-control"],
)
