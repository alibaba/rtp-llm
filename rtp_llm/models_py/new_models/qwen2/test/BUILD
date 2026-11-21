py_test(
    name = "service_variants_test",
    srcs = ["service_variants_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:model_factory",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm:transformers",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
