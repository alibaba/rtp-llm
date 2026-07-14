py_test(
    name = "generic_moe_hw_kernel_config_test",
    srcs = ["generic_moe_hw_kernel_config_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "generic_moe_unified_allreduce_test",
    srcs = ["generic_moe_unified_allreduce_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)
