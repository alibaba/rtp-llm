py_test(
    name = "test_deepseek_vl2_load",
    srcs = ["test_deepseek_vl2_load.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:deepseek_vl2_vision_loader",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_deepseek_vl2_legacy_compat",
    srcs = ["test_deepseek_vl2_load.py"],
    main = "test_deepseek_vl2_load.py",
    env = {"RTP_LLM_RUN_DEEPSEEK_VL2_LEGACY_TESTS": "1"},
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = [
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:multimodal",
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
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:deepseek_vl2_vision_loader",
    ],
)

deepseek_vl2_gpu_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm/models_py:deepseek_vl2_vision_loader",
    "//rtp_llm/models_py:new_weight_loader",
]

py_test(
    name = "test_deepseek_vl2_gpu",
    srcs = [
        "test_deepseek_vl2_gpu.py",
        "test_deepseek_vl2_load.py",
    ],
    main = "test_deepseek_vl2_gpu.py",
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = deepseek_vl2_gpu_deps,
)

py_test(
    name = "test_deepseek_vl2_rocm_gpu",
    srcs = [
        "test_deepseek_vl2_gpu.py",
        "test_deepseek_vl2_load.py",
    ],
    main = "test_deepseek_vl2_gpu.py",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = deepseek_vl2_gpu_deps,
)
