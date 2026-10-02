load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_sm100_py_test")
load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

py_library(
    name = "fused_moe_executor_test_util",
    srcs = ["fused_moe_executor_test_util.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepgemm_masked_executor_test",
    srcs = ["deepgemm_masked_executor_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "deepgemm_masked_executor_v2_cuda_graph_test",
    srcs = ["deepgemm_masked_executor_v2_cuda_graph_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    tags = ["H20_CU13"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepgemm_masked_executor_sm100_test",
    srcs = [
        "deepgemm_masked_executor_sm100_test.py",
        "deepgemm_masked_executor_test.py",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    exec_properties = {
        "gpu": "SM100_ARM_CU13",
        "gpu_count": "1",
    },
    tags = [
        "SM100_ARM_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepgemm_masked_executor_sm120_test",
    srcs = [
        "deepgemm_masked_executor_sm120_test.py",
        "deepgemm_masked_executor_test.py",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
        "gpu_count": "1",
    },
    tags = [
        "RTX_5000_PRO_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "triton_fp8_block_moe_sm120_test",
    srcs = ["triton_fp8_block_moe_sm120_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
        "gpu_count": "1",
    },
    # Intentionally runs in open-source CI: this is the SM120 correctness gate
    # for the production path, including packed UE8M0 and CUDA Graph replay.
    tags = ["RTX_5000_PRO_CU13"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "triton_fp8_block_moe_sm120_benchmark",
    srcs = ["triton_fp8_block_moe_sm120_benchmark.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
        "gpu_count": "1",
    },
    tags = [
        "RTX_5000_PRO_CU13",
        "manual",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepep_normal_executor_test",
    srcs = ["deepep_normal_executor_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepep_normal_executor_sm100_test",
    srcs = [
        "deepep_normal_executor_sm100_test.py",
        "deepep_normal_executor_test.py",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    exec_properties = {
        "gpu": "SM100_ARM_CU13",
        "gpu_count": "1",
    },
    tags = [
        "SM100_ARM_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "deepep_normal_executor_sm120_test",
    srcs = [
        "deepep_normal_executor_sm120_test.py",
        "deepep_normal_executor_test.py",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
        "gpu_count": "1",
    },
    tags = [
        "RTX_5000_PRO_CU13",
        "open_skip",
    ],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "trtllm_fp4_executor_test",
    srcs = ["trtllm_fp4_executor_test.py"],
    env = {
        "GPU_COUNT": "1",
    },
    exec_properties = {
        "gpu": "SM100_ARM_CU13",
        "gpu_count": "1",
    },
    tags = [
        "SM100_ARM_CU13",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "cutedsl_fp4_executor_test",
    srcs = ["cutedsl_fp4_executor_test.py"],
    env = {
        "GPU_COUNT": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    },
    exec_properties = {
        "gpu": "SM100_ARM_CU13",
        "gpu_count": "1",
    },
    tags = [
        "SM100_ARM_CU13",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "local_loop_test",
    srcs = ["local_loop_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

cuda13_sm100_py_test(
    name = "local_loop_cuda_test",
    srcs = ["local_loop_cuda_test.py"],
    # DeepGEMM defaults to $HOME/.deep_gemm; Bazel otherwise clears HOME.
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

cuda13_sm100_py_test(
    name = "grouped_moe_equivalence_test",
    srcs = ["grouped_moe_equivalence_test.py"],
    env_inherit = ["HOME"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

cuda13_sm100_py_test(
    name = "mega_moe_routed_only_execution_test",
    timeout = "long",
    srcs = ["mega_moe_routed_only_execution_test.py"],
    # DeepGEMM uses $HOME/.deep_gemm; reuse the existing per-user JIT cache.
    env_inherit = ["HOME"],
    gpu_count = 2,
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

cuda13_sm100_py_test(
    name = "mega_moe_end_to_end_perf_test",
    timeout = "long",
    srcs = [
        "mega_moe_end_to_end_perf_test.py",
        "mega_moe_routed_only_execution_test.py",
    ],
    env_inherit = ["HOME"],
    gpu_count = 2,
    tags = ["manual"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

cuda13_sm100_py_test(
    name = "test_grouped_fp4_strategy_perf",
    srcs = ["test_grouped_fp4_strategy_perf.py"],
    args = [
        "GroupedFP4StrategyPerfTest.test_decode_and_cuda_graph_ab_performance",
        "GroupedFP4StrategyPerfTest.test_grouped_fp4_beats_local_loop",
    ],
    env_inherit = ["HOME"],
    # Included by smoke_cuda13_arm in the regular Aone pipeline. Keep the
    # decode graph (1.10x) and prefill (0.90x) baseline ratio assertions active.
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

dsv4_sm100_py_test(
    name = "test_grouped_fp4_strategy_cuda_graph",
    src = "test_grouped_fp4_strategy_perf.py",
    args = [
        "-k",
        "test_cuda_graph_capture_bs1_matches_eager",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)
