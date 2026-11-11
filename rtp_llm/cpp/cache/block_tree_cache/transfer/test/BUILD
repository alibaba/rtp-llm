load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + cuda13_torch_link_deps + torch_deps() + no_block_copy_link_deps()

cc_library(
    name = "block_transfer_engine_test_utils",
    srcs = ["PerRankBlockTransferEngineTestUtils.cc"],
    hdrs = ["PerRankBlockTransferEngineTestUtils.h"],
    visibility = ["//rtp_llm/cpp/cache/block_tree_cache:__subpackages__"],
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_transfer_dispatcher_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = [
        "BlockTransferDispatcherTest.cc",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/legacy:TransferExecutorTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "@grpc//:grpc++",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "per_rank_transfer_engine_device_host_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = [
        "DeviceHostBufferSafetyTest.cc",
        "PerRankBlockTransferEngineDeviceHostTest.cc",
    ],
    copts = test_copts,
    tags = [
        "L20_CU13",
        "exclusive-if-local",
    ],
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "@havenask//aios/autil:env_util",
    ],
    env = {},
)

cc_test(
    name = "per_rank_transfer_engine_host_disk_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = ["PerRankBlockTransferEngineHostDiskTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "@havenask//aios/autil:env_util",
    ],
    env = {},
    tags = ["L20_CU13"],
)

cc_test(
    name = "per_rank_transfer_engine_data_correctness_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = ["PerRankBlockTransferEngineDataCorrectnessTest.cc"],
    copts = test_copts,
    tags = [
        "L20_CU13",
        "exclusive-if-local",
    ],
    deps = test_deps + [":block_transfer_engine_test_utils"],
    env = {},
)

test_suite(
    name = "block_transfer_engine_test",
    tests = [
        ":per_rank_transfer_engine_data_correctness_test",
        ":per_rank_transfer_engine_device_host_test",
        ":per_rank_transfer_engine_host_disk_test",
    ],
)

cc_test(
    name = "multi_rank_block_transfer_engine_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = ["MultiRankBlockTransferEngineTest.cc"],
    copts = test_copts,
    tags = [
        "L20_CU13",
        "exclusive-if-local",
    ],
    deps = test_deps + [
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
    ],
    env = {},
)

cc_test(
    name = "block_transfer_request_converter_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["BlockTransferRequestConverterTest.cc"],
    copts = test_copts,
    deps = test_deps + [":block_transfer_engine_test_utils"],
    env = {},
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_types_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferTypesTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_task_registry_test",
    srcs = ["TransferTaskRegistryTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_task_registry",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "transfer_completion_query_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferCompletionQueryTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_task_registry",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@grpc//:grpc++",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_completion_forget_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferCompletionForgetTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_task_registry",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@grpc//:grpc++",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "frozen_task_recovery_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["FrozenTaskRecoveryTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_task_registry",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_completion_cleanup_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferCompletionCleanupTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "frozen_task_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["FrozenTaskTest.cc"],
    copts = test_copts,
    deps = test_deps,
    tags = ["L20_CU13"],
)

cc_test(
    name = "frozen_execution_test",
    testonly = True,
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["FrozenExecutionTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
    ],
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_batch_async_context_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferBatchAsyncContextTest.cc"],
    copts = test_copts,
    deps = test_deps,
    tags = ["L20_CU13"],
)

cc_test(
    name = "transfer_stage_state_test",
    exec_properties = {"gpu": "L20_CU13"},
    srcs = ["TransferStageStateTest.cc"],
    copts = test_copts,
    deps = test_deps,
    tags = ["L20_CU13"],
)

cc_test(
    name = "per_rank_transfer_engine_crc_test",
    exec_properties = select({
        "//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13"},
        "//conditions:default": {"gpu": "H20_CU13"},
    }),
    testonly = True,
    srcs = ["PerRankBlockTransferEngineCrcTest.cc"],
    copts = test_copts,
    tags = ["exclusive-if-local"],
    deps = test_deps + [":block_transfer_engine_test_utils"],
)

cc_test(
    name = "crc_dump_writer_test",
    srcs = ["CrcDumpWriterTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:crc_dump_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "transfer_work_loop_test",
    srcs = ["TransferWorkLoopTest.cc"],
    copts = test_copts,
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "@havenask//aios/autil:env_util",
        "//rtp_llm/cpp/utils:async_profile_trace",
    ],
    exec_properties = select({
        "//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13"},
        "//conditions:default": {"gpu": "H20_CU13"},
    }),
    tags = ["exclusive-if-local"],
)

cc_test(
    name = "transfer_buffer_safety_test",
    srcs = ["TransferBufferSafetyTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_buffer_safety",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "local_buffer_safety_failure_test",
    srcs = ["LocalBufferSafetyFailureTest.cc"],
    copts = test_copts,
    deps = test_deps,
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

cc_test(
    name = "worker_transfer_registry_test",
    srcs = ["WorkerTransferRegistryTest.cc"],
    copts = test_copts,
    deps = test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test(
    name = "local_timeout_recovery_smoke_test",
    visibility = ["//visibility:public"],
    srcs = [
        "TimeoutRecoverySmokeTest.cc",
        "WorkLoopWriteQueueTimeoutSmokeTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        ":block_transfer_engine_test_utils",
        "@grpc//:grpc++",
    ],
    tags = ["L20_CU13", "smoke_case", "manual"],
    exec_properties = {"gpu": "L20_CU13"},
    timeout = "moderate",
)
