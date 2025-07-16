load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
]

test_deps = [
    "//rtp_llm/cpp:utils",
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
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },

)

cc_test(
    name = "cm2_config_test",
    srcs = [
        "Cm2ConfigTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
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

cc_test(
    name = "net_util_test",
    srcs = [
        "NetUtilTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps,
)