
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "rmsnorm_test",
    srcs = ["rmsnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fused_add_rmsnorm_test",
    srcs = ["fused_add_rmsnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fused_qkrmsnorm_test",
    srcs = ["fused_qkrmsnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "qkrmsnorm_test",
    srcs = ["qkrmsnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "group_topk_test",
    srcs = ["group_topk_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

