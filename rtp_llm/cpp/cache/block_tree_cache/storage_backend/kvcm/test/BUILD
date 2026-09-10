load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "client_wrapper_test",
    testonly = True,
    exec_properties = {"gpu": "A10"},
    srcs = [
        "ClientWrapperTest.cc",
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
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "kvcm_internal_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["KVCMInternalTest.cc"],
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

kvcm_backend_test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:client",
    "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:kvcm_storage_backend",
    "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/model_rpc:model_rpc_pool",
    "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@havenask//aios/autil:env_util",
    "@havenask//aios/autil:json",
]

cc_test_wrapper(
    name = "kvcm_local_test",
    exec_properties = {"gpu": "A10"},
    srcs = [
        "KVCMLocalTest.cc",
        "KVCMMockTestBase.h",
        "MockKVCMClient.h",
    ],
    copts = ["-fno-access-control"] + copts(),
    tags = ["exclusive-if-local"],
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = kvcm_backend_test_deps,
)

cc_test_wrapper(
    name = "kvcm_mock_full_linear_test",
    exec_properties = {"gpu": "A10"},
    srcs = [
        "KVCMMockFullLinearTest.cc",
        "KVCMMockTestBase.h",
        "MockKVCMClient.h",
    ],
    copts = ["-fno-access-control"] + copts(),
    tags = ["exclusive-if-local"],
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = kvcm_backend_test_deps,
)

cc_test_wrapper(
    name = "kvcm_mock_only_full_test",
    exec_properties = {"gpu": "A10"},
    srcs = [
        "KVCMMockOnlyFullTest.cc",
        "KVCMMockTestBase.h",
        "MockKVCMClient.h",
    ],
    copts = ["-fno-access-control"] + copts(),
    tags = ["exclusive-if-local"],
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = kvcm_backend_test_deps,
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

# Compile the actual subscriber against a recording VIP API. This exercises
# the internal branch without contacting service discovery in unit tests.
cc_test_wrapper(
    name = "vipserver_subscriber_test",
    testonly = True,
    srcs = [
        "VIPServerSubscriberTest.cc",
        "vipserver_stubs/option.h",
        "vipserver_stubs/iphost.h",
        "vipserver_stubs/vipclient.h",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:VIPServerSubscriber.cc",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:VIPServerSubscriber.h",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:Subscriber.h",
    ],
    copts = copts() + [
        "-DKVCM_INTERNAL=1",
        "-Irtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/vipserver_stubs",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:env_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "kvcm_independent_pool_test",
    testonly = True,
    srcs = ["KVCMIndependentPoolTest.cc", "KVCMMockTestBase.h", "MockKVCMClient.h"],
    copts = ["-fno-access-control"] + copts(),
    tags = ["exclusive-if-local"],
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = kvcm_backend_test_deps + [
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:subscriber",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)
