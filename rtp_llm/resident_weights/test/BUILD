py_test(
    name = "test_store",
    srcs = ["test_store.py"],
    deps = ["//rtp_llm/resident_weights"],
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "test_resident_weights",
    srcs = ["test_resident_weights.py"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm/resident_weights",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
