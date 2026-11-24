load("//:def.bzl", "cc_test_wrapper")

cc_test_wrapper(
    name = "remote_mock_state_test",
    srcs = ["RemoteMockStateTest.cpp"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_core",
        "//rtp_llm/cpp/disaggregate/cache_store/proto:remote_mock_control_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "remote_mock_timer_test",
    srcs = ["RemoteMockTimerTest.cpp"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_core",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "remote_mock_service_test",
    linkopts = ["-Wl,--wrap=_Znwm", "-Wl,--export-dynamic-symbol=*RemoteMockTimer*", "-Wl,--wrap=_ZNK7rtp_llm15RemoteMockState7resolveERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm"],
    srcs = ["RemoteMockServiceTest.cpp", "RemoteMockTestPeer.h"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_service",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cc_test_wrapper(
    name = "remote_mock_loopback_test",
    testonly = True,
    srcs = ["RemoteMockLoopbackTest.cpp"],
    data = ["//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_server"],
    deps = [
        ":remote_server_process_test_utils",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_core",
        "//rtp_llm/cpp/disaggregate/cache_store/proto:remote_mock_control_cc_proto",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest_main",
    ] + select({"@//:using_cuda13_x86": [":cuda13_torch_nvshmem"], "//conditions:default": []}),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_library(
    name = "remote_server_process_test_utils",
    testonly = True,
    hdrs = ["RemoteServerProcessTestUtils.h"],
    visibility = ["//rtp_llm:__subpackages__"],
)
