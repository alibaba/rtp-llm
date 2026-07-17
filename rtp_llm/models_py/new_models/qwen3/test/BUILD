py_test(
    name = "test_qwen3_dense_load",
    srcs = ["test_qwen3_dense_load.py"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_qwen3_base_model_integration",
    srcs = ["test_qwen3_base_model_integration.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm:numpy",
        "//rtp_llm:rtp_llm_frontend_lib",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
