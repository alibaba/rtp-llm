py_test(
    name = "kimi_k31_config_test",
    srcs = ["kimi_k31_config_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
    env = {"GPU_COUNT": "0"},
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
    timeout = "short",
)
