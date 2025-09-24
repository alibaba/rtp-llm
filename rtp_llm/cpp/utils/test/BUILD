load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()


test_deps = [
    "//rtp_llm/cpp/utils:core_utils",
    "//rtp_llm/cpp/utils:kv_cache_utils",
    "//rtp_llm/cpp/models:dfa_utils",
    "@havenask//aios/autil:json",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
]

cc_test(
    name = "lru_cache_test",
    srcs = [
        "LRUCacheTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + ["//rtp_llm/cpp/utils:lru_cache"],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },

)

cc_test(
    name = "dfa_util_test",
    srcs = [
        "DFAUtilTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)

cc_test(
    name = "prefix_to_candidate_tokens_test",
    srcs = [
        "PrefixToCandidateTokensTest.cc",
    ],
    data = [
        "gir_prefix_dict.json"
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
