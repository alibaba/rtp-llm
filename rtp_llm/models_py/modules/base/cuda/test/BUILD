py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "norm_test",
    srcs = ["norm_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "rmsresnorm_test",
    srcs = ["rmsresnorm_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "group_topk_test",
    srcs = ["group_topk_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "torch_symm_mem_test",
    srcs = ["torch_symm_mem_test.py"],
    env = {
        "TP_SIZE": "4",
        "WORLD_SIZE": "4",
    },
    exec_properties = {"gpu": "H20_CU13"},
    # Must stay on H20: symm_mem.py:81 hard-gates on device_capability < 9, so on
    # sm86 every rank disables the communicator (24 skip hits, zero coverage) and
    # the SkipTest raised inside mp.spawn surfaces as an error rather than a skip.
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/test/utils:test_util",
    ],
)

py_test(
    name = "get_topk_ragged_cp_test",
    srcs = ["get_topk_ragged_cp_test.py"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "2",
    },
    tags = ["H20_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "indexer_op_test",
    srcs = ["indexer_op_test.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = py_test_deps,
)
