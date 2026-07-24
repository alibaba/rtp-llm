py_test(
    name = "test_qwen3_vl_load",
    srcs = ["test_qwen3_vl_load.py"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_qwen3_vl_vision_loader_import",
    srcs = ["test_qwen3_vl_vision_loader_import.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py:qwen3_vl_vision_loader",
    ],
)

py_test(
    name = "test_qwen3_vl_gpu",
    srcs = ["test_qwen3_vl_gpu.py"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:qwen3_vl_vision_loader",
    ],
)

py_test(
    name = "test_qwen3_vl_rocm",
    srcs = ["test_qwen3_vl_gpu.py"],
    main = "test_qwen3_vl_gpu.py",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:qwen3_vl_vision_loader",
    ],
)
