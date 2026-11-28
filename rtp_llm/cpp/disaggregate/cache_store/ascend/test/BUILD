load("//:def.bzl", "cc_test_wrapper", "copts")

package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

cc_test = cc_test_wrapper

cc_test(
    name = "hixl_timer_queue_test",
    srcs = ["HixlTimerQueueTest.cc"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_timer_queue",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hixl_manifest_service_test",
    srcs = ["HixlManifestServiceTest.cc"],
    timeout = "short",
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_manifest_service",
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_protocol",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hixl_read_plan_test",
    srcs = ["HixlReadPlanTest.cc"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_manifest",
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_protocol",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hixl_load_scheduler_test",
    srcs = ["HixlLoadSchedulerTest.cc"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_read_worker",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pd_transfer_lease_test",
    srcs = ["TransferBlockLeaseTest.cc"],
    copts = ["-fno-access-control"] + copts(),
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_manifest",
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_read_worker",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
        "//rtp_llm/cpp/cache:transfer_block_lease",
        "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
        "//rtp_llm/models_py/bindings/core:exec_ctx_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hixl_read_progress_test",
    srcs = ["HixlReadProgressTest.cc"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_read_worker",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hixl_memory_lifetime_test",
    srcs = ["HixlMemoryLifetimeTest.cc"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/ascend:hixl_memory_util",
        "@com_google_googletest//:gtest_main",
    ],
)
