deepseek_cpu_test_deps = [
    "//rtp_llm:aiohttp",
    "//rtp_llm:config",
    "//rtp_llm:device",
    "//rtp_llm:fastapi",
    "//rtp_llm:ops",
    "//rtp_llm:psutil",
    "//rtp_llm:pydantic",
    "//rtp_llm:safetensors",
    "//rtp_llm:torch",
    "//rtp_llm:transformers",
    "//rtp_llm:utils",
    "//rtp_llm/models_py:new_weight_loader",
] + select({
    "@//:using_rocm": [
        "//rtp_llm:aiter",
        "//rtp_llm:triton",
    ],
    "//conditions:default": [],
})

deepseek_gpu_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm/models_py:new_weight_loader",
]

py_test(
    name = "test_deepseek_newloader",
    exec_properties = {"gpu": "H20"},
    srcs = ["test_deepseek_newloader.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ],
    deps = deepseek_cpu_test_deps,
)

py_test(
    name = "test_deepseek_newloader_gpu",
    srcs = [
        "test_deepseek_newloader.py",
        "test_deepseek_newloader_gpu.py",
    ],
    main = "test_deepseek_newloader_gpu.py",
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = deepseek_gpu_test_deps,
)

py_test(
    name = "test_deepseek_newloader_rocm_gpu",
    srcs = [
        "test_deepseek_newloader.py",
        "test_deepseek_newloader_gpu.py",
    ],
    main = "test_deepseek_newloader_gpu.py",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = deepseek_gpu_test_deps,
)
