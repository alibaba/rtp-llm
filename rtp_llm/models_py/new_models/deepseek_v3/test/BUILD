deepseek_test_deps = [
    "//rtp_llm:config",
    "//rtp_llm:device",
    "//rtp_llm:ops",
    "//rtp_llm:safetensors",
    "//rtp_llm:torch",
    "//rtp_llm:utils",
    "//rtp_llm/models_py:new_weight_loader",
]

py_test(
    name = "test_deepseek_newloader",
    srcs = ["test_deepseek_newloader.py"],
    deps = deepseek_test_deps,
)

py_test(
    name = "test_deepseek_newloader_gpu",
    srcs = ["test_deepseek_newloader.py"],
    main = "test_deepseek_newloader.py",
    args = [
        "DeepSeekNewloaderTest.test_moe_cuda_noaux_router_matches_reference",
        "DeepSeekNewloaderTest.test_moe_cuda_fast_select_topk_matches_reference",
        "DeepSeekNewloaderTest.test_mla_online_fused_fp8_projection_matches_bf16_reference",
        "DeepSeekNewloaderTest.test_mla_prequantized_fp8_kernel_views_execute_numerically",
    ],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["gpu"],
    deps = deepseek_test_deps,
)

py_test(
    name = "test_deepseek_newloader_rocm_gpu",
    srcs = ["test_deepseek_newloader.py"],
    main = "test_deepseek_newloader.py",
    args = [
        "DeepSeekNewloaderTest.test_moe_rocm_noaux_reference_router_matches_expected",
    ],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = [
        "gpu",
        "rocm",
    ],
    deps = deepseek_test_deps,
)
