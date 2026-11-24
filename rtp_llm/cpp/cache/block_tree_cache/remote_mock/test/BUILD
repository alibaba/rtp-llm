load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")
cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)
cc_test_wrapper(
    testonly = True,
    name = "remote_mock_adapter_test",
    srcs = ["RemoteMockAdapterTest.cc"],
    copts = copts() + ["-fno-access-control"],
    linkopts = [
        "-Wl,--wrap=_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm",
        "-Wl,--wrap=_ZN7rtp_llm23RequestBlockBufferStore20regUserBuffersScopedERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE",
        "-Wl,--wrap=_ZN7rtp_llm23RequestBlockBufferStore30setRequestBlockBufferWatchFuncERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEOSt8functionIFvbRKSt6vectorISt10shared_ptrINS_11BlockBufferEESaISD_EEEE",
        "-Wl,--wrap=_ZN7rtp_llm18RequestBlockBuffer9addBlocksERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE",
        "-Wl,--wrap=_ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
        "-Wl,--wrap=_ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE",
    ],
    tags = ["L20_CU13", "exclusive-if-local"],
    exec_properties = {"gpu": "L20_CU13"},
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:remote_copy_failure_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:remote_allocation_failure_test_utils",
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_service",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + no_block_copy_link_deps() + select({
        "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
        "//conditions:default": [],
    }),
)
cc_test_wrapper(
    testonly = True,
    name = "remote_mock_writeback_test",
    srcs = ["RemoteMockWritebackTest.cc"],
    copts = copts() + ["-fno-access-control"],
    linkopts = [
        "-Wl,--wrap=_ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
        "-Wl,--wrap=_ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE",
    ],
    tags = ["L20_CU13", "exclusive-if-local"],
    exec_properties = {"gpu": "L20_CU13"},
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:block_transfer_engine_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:remote_copy_failure_test_utils",
        "//rtp_llm/cpp/cache/block_tree_cache/transfer/test:remote_allocation_failure_test_utils",
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_service",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + no_block_copy_link_deps() + select({
        "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
        "//conditions:default": [],
    }),
)
