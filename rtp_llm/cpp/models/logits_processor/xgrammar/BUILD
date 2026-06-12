load("//:def.bzl", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xgrammar_kernels",
    srcs = ["xgrammar_kernels.cu"],
    hdrs = ["xgrammar_kernels.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
)

cc_library(
    name = "grammar_cpp_lib",
    hdrs = [
        "RtpGrammarMatcher.h",
        "XGrammarBackend.h",
    ],
    srcs = [
        "RtpGrammarMatcher.cc",
        "XGrammarBackend.cc",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
)

cc_library(
    name = "grammar_schema_validator",
    hdrs = [
        "GrammarSchemaValidator.h",
    ],
    srcs = [
        "GrammarSchemaValidator.cc",
    ],
    deps = [
        ":grammar_cpp_lib",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
    ],
    copts = copts(),
)

cc_library(
    name = "grammar_logits_processor",
    hdrs = [
        "GrammarLogitsProcessor.h",
        "ReasoningGate.h",
    ],
    srcs = [
        "GrammarLogitsProcessor.cc",
        "GrammarModule.cc",
        "ReasoningGate.cc",
    ],
    deps = [
        ":grammar_compiler",
        ":grammar_cpp_lib",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/models:sample_infos",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/cpp/models/logits_processor/xgrammar:xgrammar_kernels",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
)

# Process-level grammar compile service: worker pool + singleflight + cache.
# No scheduler / stream dependency — extracted from the former GrammarCompileGate.
cc_library(
    name = "grammar_compiler",
    hdrs = [
        "GrammarCompiler.h",
    ],
    srcs = [
        "GrammarCompiler.cc",
    ],
    deps = [
        ":grammar_cpp_lib",
        ":grammar_schema_validator",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
)

cc_library(
    name = "xgrammar_bootstrap",
    srcs = ["XGrammarBootstrap.cc"],
    hdrs = ["XGrammarBootstrap.h"],
    deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
)
