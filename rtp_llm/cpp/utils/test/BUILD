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


cc_test(
    name = "rpc_access_log_wrapper_test",
    srcs = [
        "RpcAccessLogWrapperTest.cpp",
    ],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/utils:rpc_access_log_wrapper",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_protobuf//:protobuf"],
    size = "small",
)