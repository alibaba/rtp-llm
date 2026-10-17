# only can run in cu129 so mark as manual
py_test(
    name = "sparse_mla_op_test",
    srcs = ["sparse_mla_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "sparse_mla_fp8_decode_op_test",
    srcs = ["sparse_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "flashmla_sparse_cp_op_test",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "trtllm_mla_block_table_scatter_test",
    srcs = ["trtllm_mla_block_table_scatter_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "SM103_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "SM103_CU13"},
)

py_test(
    name = "trtllm_mla_decode_domain_gate_test",
    srcs = ["trtllm_mla_decode_domain_gate_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "SM103_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "SM103_CU13"},
)
