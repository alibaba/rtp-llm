py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "decode_attn_metadata_test",
    srcs = ["decode_attn_metadata_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "sparse_attn_decode_test",
    srcs = ["sparse_attn_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "indexer_decode_test",
    srcs = ["indexer_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "kv_write_decode_test",
    srcs = ["kv_write_decode_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "forward_decode_wiring_test",
    srcs = ["forward_decode_wiring_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "attention_decode_equivalence_test",
    srcs = ["attention_decode_equivalence_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "decode_metadata_in_place_test",
    srcs = ["decode_metadata_in_place_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "decode_fmha_impl_test",
    srcs = ["decode_fmha_impl_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "compressor_decode_vectorized_test",
    srcs = ["compressor_decode_vectorized_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "indexer_decode_vectorized_test",
    srcs = ["indexer_decode_vectorized_test.py"],
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
