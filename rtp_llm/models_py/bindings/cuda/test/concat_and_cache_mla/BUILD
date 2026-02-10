
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_dpsk_bf16",
    srcs = ["test_dpsk_bf16.py", "util.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_dpsk32_fp8",
    srcs = ["test_dpsk32_fp8.py", "util.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_model1_fp8",
    srcs = ["test_model1_fp8.py", "util.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)