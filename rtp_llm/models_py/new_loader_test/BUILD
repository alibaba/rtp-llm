py_test(
    name = "foundation_test",
    srcs = ["foundation_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "foundation_gpu_test",
    srcs = ["foundation_gpu_test.py"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["gpu"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)
