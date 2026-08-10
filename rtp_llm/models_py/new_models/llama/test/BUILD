py_test(
    name = "test_llama_load",
    srcs = ["test_llama_load.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

llama_gpu_test_deps = [
    "//rtp_llm:torch",
    "//rtp_llm/models_py:new_weight_loader",
]

py_test(
    name = "test_llama_gpu",
    srcs = ["test_llama_gpu.py"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = llama_gpu_test_deps,
)

py_test(
    name = "test_llama_rocm_gpu",
    srcs = ["test_llama_gpu.py"],
    main = "test_llama_gpu.py",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = llama_gpu_test_deps,
)
