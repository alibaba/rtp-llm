py_test(
    name = "dsv4_kv_cache_test",
    srcs = ["dsv4_kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "qwen3_reranker_test",
    srcs = ["qwen3_reranker_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["RTX_5000_PRO", "RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)
