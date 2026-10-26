load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

google_test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
]

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
    name = "linear_blocks_utils_test",
    srcs = [
        "LinearBlocksUtilsTest.cc",
    ],
    data = [],
    deps = google_test_deps + ["//rtp_llm/cpp/utils:linear_blocks_util"],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

cc_test(
    name = "signal_utils_test",
    srcs = ["SignalUtilsTest.cc"],
    copts = test_copts,
    deps = google_test_deps + [
        "//rtp_llm/cpp/utils:signal_utils",
    ] + cuda13_torch_link_deps,
)
