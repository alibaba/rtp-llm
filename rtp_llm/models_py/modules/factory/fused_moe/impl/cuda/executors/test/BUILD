load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_sm100_py_test")

py_library(
    name = "fused_moe_executor_test_util",
    srcs = ["fused_moe_executor_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "deepgemm_masked_executor_test",
    srcs = ["deepgemm_masked_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepgemm_masked_executor_v2_cuda_graph_test",
    srcs = ["deepgemm_masked_executor_v2_cuda_graph_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepgemm_masked_executor_sm100_test",
    srcs = ["deepgemm_masked_executor_test.py",
            "deepgemm_masked_executor_sm100_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "deepep_normal_executor_test",
    srcs = ["deepep_normal_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepep_normal_executor_sm100_test",
    srcs = ["deepep_normal_executor_sm100_test.py",
            "deepep_normal_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "trtllm_fp4_executor_test",
    srcs = ["trtllm_fp4_executor_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "cutedsl_fp4_executor_test",
    srcs = ["cutedsl_fp4_executor_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {
        "GPU_COUNT": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test(
    name = "local_loop_test",
    srcs = ["local_loop_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

cuda13_sm100_py_test(
    name = "local_loop_cuda_test",
    srcs = ["local_loop_cuda_test.py"],
    # DeepGEMM defaults to $HOME/.deep_gemm; Bazel otherwise clears HOME.
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

cuda13_sm100_py_test(
    name = "grouped_moe_equivalence_test",
    srcs = ["grouped_moe_equivalence_test.py"],
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

cuda13_sm100_py_test(
    name = "mega_moe_routed_only_execution_test",
    srcs = ["mega_moe_routed_only_execution_test.py"],
    # DeepGEMM uses $HOME/.deep_gemm; reuse the existing per-user JIT cache.
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    gpu_count = 2,
)

cuda13_sm100_py_test(
    name = "mega_moe_end_to_end_perf_test",
    srcs = [
        "mega_moe_end_to_end_perf_test.py",
        "mega_moe_routed_only_execution_test.py",
    ],
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    timeout = "long",
    tags = ["manual"],
    gpu_count = 2,
)

cuda13_sm100_py_test(
    name = "test_grouped_fp4_strategy_perf",
    srcs = ["test_grouped_fp4_strategy_perf.py"],
    env_inherit = ["HOME"],
    args = [
        "GroupedFP4StrategyPerfTest.test_decode_and_cuda_graph_ab_performance",
        "GroupedFP4StrategyPerfTest.test_grouped_fp4_beats_local_loop",
    ],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    # Included by smoke_cuda13_arm in the regular Aone pipeline. Keep the
    # decode graph (1.10x) and prefill (0.90x) baseline ratio assertions active.
    visibility = ["//visibility:public"],
)

dsv4_sm100_py_test(
    name = "test_grouped_fp4_strategy_cuda_graph",
    src = "test_grouped_fp4_strategy_perf.py",
    args = ["-k", "test_cuda_graph_capture_bs1_matches_eager"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//visibility:public"],
)
