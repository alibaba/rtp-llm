load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "client_wrapper_lifecycle_test",
    exec_properties = {"gpu": "A10"},
    srcs = [
        "ClientWrapperLifecycleTest.cc",
        "MockKVCMClient.h",
    ],
    copts = copts(),
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:client",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:subscriber",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "group_policy_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["GroupPolicyTest.cc"],
    copts = copts(),
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:group_policy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "kvcm_storage_backend_test",
    exec_properties = {"gpu": "A10"},
    srcs = [
        "KVCMStorageBackendTest.cc",
        "MockKVCMClient.h",
    ],
    copts = copts(),
    tags = ["exclusive-if-local"],
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:client",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:kvcm_storage_backend",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:json",
    ],
)

cc_test_wrapper(
    name = "kvcm_config_test",
    srcs = ["KVCMConfigTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:json",
    ],
)
