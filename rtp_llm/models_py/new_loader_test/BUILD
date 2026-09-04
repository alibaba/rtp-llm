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
    tags = ["H20"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "multi_gpu_device_context_test",
    srcs = ["multi_gpu_device_context_test.py"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "2",
    },
    tags = ["H20"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "rocm_odd_rmsnorm_test",
    srcs = ["rocm_odd_rmsnorm_test.py"],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)
