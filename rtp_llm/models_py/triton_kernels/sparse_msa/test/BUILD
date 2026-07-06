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
