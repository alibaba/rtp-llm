deepseek_test_deps = [
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

py_test(
    name = "test_deepseek_newloader",
    srcs = ["test_deepseek_newloader.py"],
    main = "test_deepseek_newloader.py",
    deps = deepseek_test_deps,
)

py_test(
    name = "test_deepseek_newloader_gpu",
    srcs = ["test_deepseek_newloader.py"],
    main = "test_deepseek_newloader.py",
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = deepseek_test_deps,
)

py_test(
    name = "test_deepseek_newloader_rocm_gpu",
    srcs = ["test_deepseek_newloader.py"],
    main = "test_deepseek_newloader.py",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = ["rocm"],
    deps = deepseek_test_deps,
)
