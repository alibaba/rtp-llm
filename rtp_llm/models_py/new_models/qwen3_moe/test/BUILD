py_test(
    name = "test_qwen3_moe_load",
    exec_properties = {"gpu": "H20"},
    srcs = ["test_qwen3_moe_load.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
