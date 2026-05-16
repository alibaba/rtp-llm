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
