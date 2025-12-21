
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "fused_moe_executor_benchmark",
    srcs = ["fused_moe_executor_benchmark.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "manual"],
    exec_properties = {'gpu':'H20'},
)
