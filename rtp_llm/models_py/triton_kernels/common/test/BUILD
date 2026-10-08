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
    # Not sm90-bound: the only hard constraint is Triton's fp8e4nv storage type,
    # which needs compute capability >= 89. A10 (sm86) would hard-fail with
    # "type fp8e4nv not supported in this architecture", so this runs on L20
    # (sm89) instead of H20. The H20 latency benchmark in this file is not
    # collected as a unit test.
    tags = ["L20_CU13"],
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
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
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = ["A10_CU13"],
)

py_test(
    name = "situ_and_mul_test",
    srcs = ["situ_and_mul_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {'gpu':'L20', 'gpu_count':'1'},
)

py_test(
    name = "scatter_qkv_test",
    srcs = ["scatter_qkv_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = ["A10_CU13"],
)
