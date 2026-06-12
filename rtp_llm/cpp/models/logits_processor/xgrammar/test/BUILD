load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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
    srcs = ["XGrammarBackendTest.cc"],
    deps = test_deps + [
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_cpp_lib",
    ],
    copts = copts() + ["-fno-access-control"],
)

cc_test(
    name = "grammar_logits_processor_test",
    srcs = ["GrammarLogitsProcessorTest.cc"],
    deps = test_deps + [
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_cpp_lib",
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_logits_processor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = copts(),
)

cc_test(
    name = "grammar_schema_validator_test",
    srcs = ["GrammarSchemaValidatorTest.cc"],
    deps = test_deps + [
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_schema_validator",
    ],
    copts = copts(),
)

cc_test(
    name = "grammar_compiler_test",
    srcs = ["GrammarCompilerTest.cc"],
    data = [],
    copts = copts() + ["-fno-access-control"],
    deps = test_deps + [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_cpp_lib",
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_compiler",
        "//rtp_llm/cpp/models/logits_processor/xgrammar:grammar_logits_processor",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
