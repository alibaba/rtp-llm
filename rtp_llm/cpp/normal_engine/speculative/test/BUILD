load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "mtp_deferred_sampler_test",
    srcs = ["MtpBatchStreamProcessorTest.cc"],
    data = ["//rtp_llm/models_py/sampling:sampling"],
    copts = test_copts + ["-DRTP_TEST_DEFERRED_SAMPLING"],
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/models:logits_processor",
    ] + torch_deps() + cuda13_torch_link_deps,
    args = ["--gtest_filter=*DeferredTargetSamplingEligibility"],
    env = {"TEST_USING_DEVICE": "CUDA", "RTP_LLM_TEST_PYTHON_SAMPLING": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

# Separate processes are intentional: runtime optimization flags are cached.
[
    cc_test(
        name = "mtp_executor_overlap_fusion_%s_test" % fusion,
        srcs = ["MtpExecutorTest.cc"],
        copts = test_copts,
        deps = [
            "//rtp_llm/models_py/bindings/cuda/kernels:mtp_token_gather",
            "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
            "//rtp_llm/models_py/bindings/cuda/kernels:mtp_target_verify_prepare",
            "//rtp_llm/cpp/testing:device_test_utils",
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
            "//rtp_llm/cpp/normal_engine:normal_engine",
            "//rtp_llm/cpp/utils:mtp_fusion",
            "//rtp_llm/cpp/models:models",
            "//rtp_llm/cpp/config:config_modules",
            "//rtp_llm/cpp/normal_engine/test:mock_engine",
            "@com_google_googletest//:gtest",
            "@com_google_googletest//:gtest_main",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            "//rtp_llm/cpp/engine_base/schedulers:schedulers",
            "//rtp_llm/cpp/config:static_config",
        ] + torch_deps() + cuda13_torch_link_deps,
        args = ["--gtest_filter=*ForwardOverlap*"],
        env = {
            "TEST_USING_DEVICE": "CUDA",
            "RTP_LLM_MTP_OPTIMIZE": fusion,
            "RTP_LLM_MTP_FORWARD_OVERLAP": "1",
            "RTP_LLM_STREAM_ASYNC": "1",
            "RTP_LLM_DEVICE_INPUT": "1",
            "RTP_LLM_MTP_ASYNC_PREPARE": "0",
        },
        exec_properties = {"gpu": "H20"},
    )
    for fusion in ["0", "1"]
]

cc_test(
    name = "mtp_batch_stream_processor_fusion_test",
    srcs = ["MtpBatchStreamProcessorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/models:logits_processor",
    ] + torch_deps() + cuda13_torch_link_deps,
    args = ["--gtest_filter=*DecodeForwardDynamicBatch*:*DispatchSnapshot*"],
    env = {"TEST_USING_DEVICE": "CUDA", "RTP_LLM_MTP_OPTIMIZE": "1"},
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "mtp_batch_stream_processor_test",
    srcs = glob([
        "MtpBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "mtp_executor_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels:mtp_token_gather",
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
        "//rtp_llm/models_py/bindings/cuda/kernels:mtp_target_verify_prepare",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "mtp_profile_window_sm103_test",
    srcs = ["MtpExecutorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels:mtp_token_gather",
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
        "//rtp_llm/models_py/bindings/cuda/kernels:mtp_target_verify_prepare",
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/config:static_config",
    ] + torch_deps() + cuda13_torch_link_deps,
    args = ["--gtest_filter=MtpExecutorTest.testProfileWindowSkipsIdleAndClosesOnPrefillFailure:MtpExecutorTest.testSingleBatchPrefill:MtpExecutorTest.testSingleBatchDecode"],
    env = {"TEST_USING_DEVICE": "CUDA"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

cc_test(
    name = "mtp_prefill_target_token_sm103_test",
    srcs = ["MtpBatchStreamProcessorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/utils:mtp_fusion",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/models:logits_processor",
    ] + torch_deps() + cuda13_torch_link_deps,
    args = ["--gtest_filter=*PrefillDispatch*:*PrefillTargetLogprobs*:*PrefillThinkingToken*"],
    env = {"TEST_USING_DEVICE": "CUDA"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

# Keep the real overlap-path regressions enabled without shell overrides.
cc_test(
    name = "mtp_forward_overlap_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/config:static_config",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "RTP_LLM_MTP_FORWARD_OVERLAP": "1",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DEVICE_INPUT": "1",
        "RTP_LLM_MTP_ASYNC_PREPARE": "0",
        "GTEST_FILTER": "MtpExecutorTest.testForwardOverlap*",
    },
    exec_properties = {'gpu':'H20'},
)
