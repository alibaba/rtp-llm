load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

device_runtime_deps = device_impl_target() + select({
    "@//:using_cuda": [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "mock_engine",
    hdrs = [
        "MockEngine.h",
    ],
    srcs = [
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/engine_base:weights_converter",
        "//rtp_llm/cpp/models",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + device_runtime_deps + cuda13_torch_link_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "normal_output_dispatcher_test",
    srcs = ["NormalOutputDispatcherTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test(
    name = "batch_stream_processor_test",
    srcs = glob([
        "NormalBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/models",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + device_runtime_deps + cuda13_torch_link_deps,
    env = device_test_envs(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test(
    name = "engine_test",
    srcs = glob([
        "EngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        ":mock_engine",
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/engine_base/schedulers",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = device_test_envs(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_library(
    name = "tail_reuse_cp_completion_integration_test_lib",
    srcs = ["TailReuseCpCompletionIntegrationTest.cc"],
    alwayslink = True,
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/cache/block_tree_cache:block_tree_cache",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/models",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    ] + torch_deps() + device_runtime_deps + cuda13_torch_link_deps,
)

cc_binary(
    name = "th_tail_reuse_cp_completion_integration_test",
    deps = [":tail_reuse_cp_completion_integration_test_lib"],
    linkshared = 1,
)

py_test(
    name = "tail_reuse_cp_completion_integration_test",
    srcs = ["tail_reuse_cp_completion_integration_test.py"],
    data = [":th_tail_reuse_cp_completion_integration_test"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
    ],
    env = {"GPU_COUNT": "2", "TEST_USING_DEVICE": "CUDA"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "2"},
    tags = ["L20_CU13"],
    timeout = "moderate",
)

py_test(
    name = "tail_reuse_tp_subgroups_integration_test",
    srcs = ["tail_reuse_cp_completion_integration_test.py"],
    main = "tail_reuse_cp_completion_integration_test.py",
    data = [":th_tail_reuse_cp_completion_integration_test"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
    ],
    env = {"GPU_COUNT": "4", "TEST_USING_DEVICE": "CUDA", "TAIL_REUSE_TP_GROUP_COUNT": "2"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "4"},
    tags = ["L20_CU13"],
    timeout = "moderate",
)

cc_test(
    name = "async_runner_test",
    srcs = [
        "AsyncRunnerTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + device_runtime_deps + cuda13_torch_link_deps,
    env = device_test_envs(),
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

# The executor caches RTP_LLM_STREAM_ASYNC, so test both settings in fresh processes.
[
    cc_test(
        name = "engine_stop_{}_test".format(mode),
        srcs = ["EngineStopTest.cc"],
        copts = test_copts,
        env = select({
            "@//:using_rocm": {
                "TEST_USING_DEVICE": "ROCM",
                "RTP_LLM_STREAM_ASYNC": stream_async,
            },
            "//conditions:default": {
                "TEST_USING_DEVICE": "CUDA",
                "LD_PRELOAD": "libtorch_cpu.so",
                "RTP_LLM_STREAM_ASYNC": stream_async,
            },
        }),
        exec_properties = {"gpu": "L20_CU13"},
        tags = ["L20_CU13"],
        deps = [":mock_engine"] + cuda13_torch_link_deps,
    )
    for mode, stream_async in [
        ("sync", "0"),
        ("async", "1"),
    ]
]
