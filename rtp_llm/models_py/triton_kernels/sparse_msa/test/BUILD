py_test(
    name = "test_minimax_decode_topk",
    srcs = ["test_minimax_decode_topk.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
)

# Chunked (M3_SPARSE_ATTN_CHUNK_ENABLE) vs non-chunked sparse attention
# precision regression test. Runs on SM100 (fmha_sm100 cute-DSL backend).
py_test(
    name = "test_sparse_prefill_chunk",
    srcs = ["test_sparse_prefill_chunk.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)
