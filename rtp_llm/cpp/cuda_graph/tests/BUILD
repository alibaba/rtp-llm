load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

cc_test(
    name = "gdn_decode_state_validation_test",
    srcs = ["gdn_decode_state_validation_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:gdn_decode_state_validation",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

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
