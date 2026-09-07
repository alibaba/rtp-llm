py_test (
    name = "silu_mul_masked_test",
    srcs = ["silu_mul_masked_test.py"],
    deps = [
        "//rtp_llm:matplotlib",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test(
    name = "silu_mul_masked_heuristic_test",
    srcs = ["silu_mul_masked_heuristic_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_moe_gating",
    srcs = ["test_moe_gating.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "scatter_qkv_test",
    srcs = ["scatter_qkv_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "silu_mul_masked_packed_sm120_test",
    srcs = ["silu_mul_masked_packed_sm120_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO", "gpu_count": "1"},
)

py_test(
    name = "silu_mul_masked_packed_sm120_benchmark",
    srcs = ["silu_mul_masked_packed_sm120_benchmark.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "open_skip", "RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO", "gpu_count": "1"},
)
