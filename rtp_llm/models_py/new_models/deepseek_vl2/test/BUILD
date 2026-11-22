py_test(
    name = "test_deepseek_vl2_load",
    srcs = ["test_deepseek_vl2_load.py"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm:multimodal",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py/new_models/glm4_moe/test:checkpoint_test_support",
    ],
)
