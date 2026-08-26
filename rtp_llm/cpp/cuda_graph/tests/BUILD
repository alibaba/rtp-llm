load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "cuda13_test_exec_properties", "torch_deps")
load("@rules_cc//cc:defs.bzl", cuda13_cc_test = "cc_test")

cuda13_cc_test(
    name = "cuda_graph_replay_retry_integration_test",
    srcs = ["cuda_graph_replay_retry_integration_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/utils:torch_cuda_oom",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = cuda13_test_exec_properties(),
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    timeout = "short",
)

cc_test(
    name = "combo_position_ids_validation_test",
    srcs = ["combo_position_ids_validation_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:combo_position_ids_validation",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "cuda_graph_copy_kernel_test_rocm",
    main = "cuda_graph_copy_kernel_test.py",
    srcs = ["cuda_graph_copy_kernel_test.py"],
    data = ["//:th_transformer"],
    deps = ["//rtp_llm/test/model_test/test_util:test_util"],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "ROCM",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
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
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
        "@com_google_googletest//:gtest_main",
    ],
    tags = ["rocm"],
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "cuda_graph_test_runner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "test_cuda_graph_runner",
    deps = [
        ":test_cuda_graph_runner_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_library(
    name = "cuda_graph_test_utils",
    srcs = ["cuda_graph_test_utils.py"],
    deps = [
        "//rtp_llm/models_py:utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "cuda_graph_replay_retry_test",
    srcs = ["cuda_graph_replay_retry_test.cc"],
    deps = [
        "//rtp_llm/cpp/utils:torch_cuda_oom",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + cuda13_torch_link_deps,
    copts = copts(),
    timeout = "short",
)

cc_test(
    name = "cuda_graph_replay_retry_integration_test",
    srcs = ["cuda_graph_replay_retry_integration_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/cpp/utils:torch_cuda_oom",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "H20"},
    timeout = "short",
)

cc_test(
    name = "cuda_graph_replay_real_oom_manual_test",
    srcs = ["cuda_graph_replay_real_oom_manual_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/cpp/utils:torch_cuda_oom",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    },
    exec_properties = {"gpu": "H20"},
    tags = ["manual"],
    timeout = "moderate",
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
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    exec_properties = {'gpu':'A10'},
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
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "LOAD_METHOD": "scratch",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "SEQ_SIZE_PER_BLOCK": "64",
        "ACT_TYPE": "BF16",
    },
    tags = ["H20"],
    # Back on H20 after measuring on sm86. Passes on H20 (ut-sm9x, run
    # 69503829); on sm86 it fails deterministically, 2 of 2, during unittest
    # collection: py_flashinfer_mha.py:369 raises "kv_cache is required for
    # paged attention" from __init__ -> init_prefill. A different fmha backend
    # is selected below sm90 and this fixture does not supply a kv_cache for it.
    # Which selection branch diverges is not yet pinned down -- the is_sm90()
    # gate at py_flashinfer_mha.py:135 is FP8-Q-only, so it is not that one.
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "cuda_graph_tagged_cache_test",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "cuda_graph_tagged_cache_test_rocm",
    main = "cuda_graph_tagged_cache_test.py",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "ROCM",
    },
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "cuda_graph_dirty_capture_test",
    main = "cuda_graph_tagged_cache_test.py",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    args = ["TestCudaGraphTaggedCache.test_dirty_capture_failure_is_fail_closed"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "CUDA",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "cuda_graph_dirty_capture_test_rocm",
    main = "cuda_graph_tagged_cache_test.py",
    srcs = ["cuda_graph_tagged_cache_test.py"],
    args = ["TestCudaGraphTaggedCache.test_dirty_capture_failure_is_fail_closed"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        ":cuda_graph_test_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
        "NOT_USE_DEFAULT_STREAM": "1",
        "RTP_LLM_RUN_DIRTY_CAPTURE_TEST": "1",
        "TEST_USING_DEVICE": "ROCM",
    },
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "cuda_graph_copy_kernel_test",
    srcs = [
        "cuda_graph_copy_kernel_test.py",
    ],
    data = [
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "cuda_graph_compiled_warmup_test",
    srcs = ["cuda_graph_compiled_warmup_test.py"],
    data = [
        ":test_cuda_graph_runner",
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)
