py_test(
    name = "shared_layers_test",
    srcs = ["shared_layers_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    },
    deps = [
        "//rtp_llm/models_py:new_loader_core",
        "//rtp_llm/models_py:new_loader_layers",
        "//rtp_llm/models_py:new_loader_quant_methods",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)
