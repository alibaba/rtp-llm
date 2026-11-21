py_test(
    name = "test_kimi_gpu",
    srcs = ["test_kimi_gpu.py", "test_kimi_vision.py", "test_kimi_compressed.py"],
    tags = ["H20", "H20_CU13"],
    exec_properties = select({
        "//:using_cuda13_x86": {"gpu": "H20_CU13"},
        "//conditions:default": {"gpu": "H20"},
    }),
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_language_native_forward",
    srcs = ["test_language_native_forward.py", "test_language.py"],
    # The fixture uses CPU tensors, but PyModelOutputs needs the compute DSO.
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    data = ["//rtp_llm/libs:librtp_compute_ops_so"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_language",
    srcs = ["test_language.py"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_kimi_vision",
    srcs = ["test_kimi_vision.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_kimi_compressed",
    srcs = ["test_kimi_compressed.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
