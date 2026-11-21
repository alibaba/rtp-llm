py_test(
    name = "test_kda_parameters",
    srcs = ["test_kda_parameters.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_kimi_owner",
    srcs = ["test_kimi_owner.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)
