py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "indexer_decode_test",
    srcs = ["indexer_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "fp8_kv_quant_decode_test",
    srcs = ["fp8_kv_quant_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "fp8_sparse_attn_decode_test",
    srcs = ["fp8_sparse_attn_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)
