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
    srcs = ["BlockTransferDispatcherTest.cc"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "per_rank_transfer_engine_device_host_test",
    testonly = True,
    srcs = ["PerRankBlockTransferEngineDeviceHostTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [":block_transfer_engine_test_utils"],
    env = {},
)

cc_test(
    name = "per_rank_transfer_engine_host_disk_test",
    testonly = True,
    srcs = ["PerRankBlockTransferEngineHostDiskTest.cc"],
    copts = test_copts,
    deps = test_deps + [":block_transfer_engine_test_utils"],
    env = {},
)

cc_test(
    name = "per_rank_transfer_engine_data_correctness_test",
    testonly = True,
    srcs = ["PerRankBlockTransferEngineDataCorrectnessTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
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
    testonly = True,
    srcs = ["MultiRankBlockTransferEngineTest.cc"],
    copts = test_copts,
    tags = ["exclusive"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
    ],
    env = {},
)

cc_test(
    name = "block_transfer_request_converter_test",
    srcs = ["BlockTransferRequestConverterTest.cc"],
    copts = test_copts,
    deps = test_deps + [":block_transfer_engine_test_utils"],
    env = {},
)

cc_test(
    name = "transfer_types_test",
    srcs = ["TransferTypesTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
)

cc_test(
    name = "transfer_batch_async_context_test",
    srcs = ["TransferBatchAsyncContextTest.cc"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "transfer_stage_state_test",
    srcs = ["TransferStageStateTest.cc"],
    copts = test_copts,
    deps = test_deps,
)
