package(default_visibility = ["//visibility:public"])

py_library(
    name = "fp8_per_tensor_test_support",
    srcs = ["test_fp8_per_tensor.py"],
    deps = [
        "//rtp_llm/models_py:new_loader_core",
        "//rtp_llm/models_py:new_loader_layers",
        "//rtp_llm/models_py:new_loader_quant_methods",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_fp8_per_tensor",
    srcs = ["test_fp8_per_tensor.py"],
    main = "test_fp8_per_tensor.py",
    deps = [":fp8_per_tensor_test_support"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_fp8_per_tensor_gpu",
    srcs = ["test_fp8_per_tensor_gpu.py"],
    deps = [":fp8_per_tensor_test_support"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    env = {"GPU_COUNT": "1"},
)

py_test(
    name = "test_fp8_per_tensor_gpu_rocm",
    srcs = ["test_fp8_per_tensor_gpu.py"],
    main = "test_fp8_per_tensor_gpu.py",
    deps = [":fp8_per_tensor_test_support"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    env = {"GPU_COUNT": "1"},
)
