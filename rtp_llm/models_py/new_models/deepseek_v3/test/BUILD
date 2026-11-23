load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_test(
    name = "test_mla_ft_roundtrip",
    srcs = ["test_mla_ft_roundtrip.py", "test_glm5_mtp_route.py", "test_nextn_route.py", "test_score_route.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch", "//rtp_llm/tools/convert:convert"],
)

py_test(
    name = "test_mla_ft_gpu",
    srcs = ["test_mla_ft_gpu.py", "test_mla_ft_roundtrip.py", "test_glm5_mtp_route.py", "test_nextn_route.py", "test_score_route.py"],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch", "//rtp_llm/tools/convert:convert"],
)

py_test(
    name = "test_sparse_indexer_checkpoint",
    srcs = ["test_sparse_indexer_checkpoint.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_score_checkpoint",
    srcs = ["test_score_checkpoint.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "test_score_route",
    srcs = ["test_score_route.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_mla_checkpoint_route",
    srcs = ["test_mla_checkpoint_route.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_mla_quantized_checkpoint",
    srcs = ["test_mla_quantized_checkpoint.py"],
    env = {"OMP_NUM_THREADS": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_mla_quantized_checkpoint_sm100",
    srcs = ["test_mla_quantized_checkpoint.py"],
    main = "test_mla_quantized_checkpoint.py",
    args = ["QuantizedMlaTest.test_blackwell_real_fused_fp8_forward"],
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_mla_rope_config",
    srcs = ["test_mla_rope_config.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_nextn_route",
    srcs = ["test_nextn_route.py", "test_score_route.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_glm5_mtp_route",
    srcs = ["test_glm5_mtp_route.py", "test_nextn_route.py", "test_score_route.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "test_glm5_mtp_native_output",
    srcs = ["test_glm5_mtp_native_output.py", "test_glm5_mtp_route.py", "test_nextn_route.py", "test_score_route.py"],
    env = checkpoint_reader_test_env(),
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    data = ["//rtp_llm/libs:librtp_compute_ops_so"],
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)
