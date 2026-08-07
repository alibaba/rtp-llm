py_test(
    name = "test_deepseek_vl2_load",
    srcs = ["test_deepseek_vl2_load.py"],
    deps = [
        "//rtp_llm:multimodal",
        "//rtp_llm:models",
        "//rtp_llm:lora",
        "//rtp_llm:safetensors",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:deepseek_vl2_vision_loader",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_deepseek_vl2_vision_loader_import",
    srcs = ["test_deepseek_vl2_vision_loader_import.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py:deepseek_vl2_vision_loader",
    ],
)
