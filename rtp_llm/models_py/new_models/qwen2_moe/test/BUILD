py_test(
    name = "test_qwen2_moe_load",
    exec_properties = {"gpu": "H20"},
    srcs = ["test_qwen2_moe_load.py"],
    tags = ["H20"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
