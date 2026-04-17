
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "norm_test",
    srcs = ["norm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "rmsresnorm_test",
    srcs = ["rmsresnorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "group_topk_test",
    srcs = ["group_topk_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)


py_test (
    name = "torch_symm_mem_test",
    srcs = ["torch_symm_mem_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/test/utils:test_util",
    ],
    env = {
        "TP_SIZE": "4",
        "WORLD_SIZE": "4",
    },
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "get_topk_ragged_cp_test",
    srcs = ["get_topk_ragged_cp_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)