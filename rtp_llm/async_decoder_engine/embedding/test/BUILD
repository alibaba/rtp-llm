py_test(
    name = "embedding_engine_graph_gate_test",
    srcs = ["embedding_engine_graph_gate_test.py"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
        "gpu_count": "1",
    },
    deps = [
        "//rtp_llm:testlib",
    ],
)
