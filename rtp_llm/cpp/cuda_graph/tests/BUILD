load("@arch_config//:arch_select.bzl", "torch_deps")
load("@rules_cc//cc:defs.bzl", cuda13_cc_test = "cc_test")
load("//:def.bzl", "copts")

[
    cuda13_cc_test(
        name = "cuda_graph_replay_retry_integration_test_" + platform["suffix"],
        timeout = "short",
        srcs = ["cuda_graph_replay_retry_integration_test.cc"],
        copts = copts(),
        exec_properties = {"gpu": platform["gpu"]},
        tags = [platform["gpu"]],
        target_compatible_with = select({
            platform["config"]: [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        deps = [
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
            "//rtp_llm/cpp/utils:torch_cuda_oom",
            "@com_google_googletest//:gtest_main",
        ] + torch_deps(),
    )
    for platform in [
        {
            "config": "//:using_cuda13_arm",
            "gpu": "SM100_ARM_CU13",
            "suffix": "arm",
        },
        {
            "config": "//:using_cuda13_x86",
            "gpu": "L20D_TEST",
            "suffix": "x86",
        },
    ]
]

cc_test(
    name = "combo_position_ids_validation_test",
    srcs = ["combo_position_ids_validation_test.cc"],
    copts = copts(),
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/cpp/cuda_graph:combo_position_ids_validation",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

py_test(
    name = "cuda_graph_copy_kernel_test_rocm",
    srcs = ["cuda_graph_copy_kernel_test.py"],
    data = ["//:th_transformer"],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "ROCM",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "cuda_graph_copy_kernel_test.py",
    tags = ["rocm"],
    deps = ["//rtp_llm/test/model_test/test_util"],
)

cc_test(
    name = "prepared_attention_inputs_guard_test",
    srcs = ["prepared_attention_inputs_guard_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:prepared_attention_inputs_guard",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "generation_prefill_cuda_graph_replay_metadata_test",
    srcs = ["generation_prefill_cuda_graph_replay_metadata_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:generation_prefill_cuda_graph_replay_metadata",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_graph_prepare_fill_rocm_test",
    srcs = ["cuda_graph_prepare_fill_test.cc"],
    copts = copts(),
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "cuda_graph_test_runner.cc",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    alwayslink = True,
)

cc_binary(
    name = "test_cuda_graph_runner",
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":test_cuda_graph_runner_libs",
    ],
)

py_library(
    name = "cuda_graph_test_utils",
    srcs = ["cuda_graph_test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py:utils",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_decode_padding",
    srcs = [
        "cuda_graph_decode_padding.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_prefill",
    srcs = [
        "cuda_graph_prefill.py",
    ],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "SEQ_SIZE_PER_BLOCK": "64",
        "ACT_TYPE": "BF16",
    },
    # Back on H20 after measuring on sm86. Passes on H20 (ut-sm9x, run
    # 69503829); on sm86 it fails deterministically, 2 of 2, during unittest
    # collection: py_flashinfer_mha.py:369 raises "kv_cache is required for
    # paged attention" from __init__ -> init_prefill. A different fmha backend
    # is selected below sm90 and this fixture does not supply a kv_cache for it.
    # Which selection branch diverges is not yet pinned down -- the is_sm90()
    # gate at py_flashinfer_mha.py:135 is FP8-Q-only, so it is not that one.
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_tagged_cache_test",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_tagged_cache_test_rocm",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "ROCM",
    },
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "cuda_graph_tagged_cache_test.py",
    tags = ["rocm"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_dirty_capture_test",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    args = ["TestCudaGraphTaggedCache.test_dirty_capture_failure_is_fail_closed"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "H20_CU13"},
    main = "cuda_graph_tagged_cache_test.py",
    tags = ["H20_CU13"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_dirty_capture_test_rocm",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    args = ["TestCudaGraphTaggedCache.test_dirty_capture_failure_is_fail_closed"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "ROCM",
    },
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "cuda_graph_tagged_cache_test.py",
    tags = ["rocm"],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_copy_kernel_test",
    srcs = [
        "cuda_graph_copy_kernel_test.py",
    ],
    data = [
        "//:th_transformer",
    ],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "cuda_graph_compiled_warmup_test",
    srcs = ["cuda_graph_compiled_warmup_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
    tags = ["RTX_5000_PRO_CU13"],
    deps = [
        "//rtp_llm/test/model_test/test_util",
    ],
)
