load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_test(
    name = "test_fp8_online_scale_layout",
    srcs = ["test_fp8_online_scale_layout.py"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors"],
    env = checkpoint_reader_test_env(),
)

py_library(
    name = "mxfp4_test_fixtures",
    srcs = ["test_mxfp4_quark.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_mxfp4_quark",
    srcs = ["test_mxfp4_quark.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "test_fp8_already_quantized",
    srcs = ["test_fp8_already_quantized.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "rocm_ptpc_weight_encoding_test",
    srcs = ["rocm_ptpc_weight_encoding_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "bert_ptpc_epilogue_test",
    srcs = ["bert_ptpc_epilogue_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "bert_ptpc_epilogue_gpu_test",
    srcs = ["bert_ptpc_epilogue_gpu_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_fp8_cpu_load",
    srcs = ["test_fp8_cpu_load.py"],
    env = checkpoint_reader_test_env(),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
        "//rtp_llm:safetensors",
    ],
)

py_test(
    name = "test_int8_ft_export",
    srcs = ["test_int8_ft_export.py"],
    env = checkpoint_reader_test_env({}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py:int8_export_helpers",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/tools/convert:convert",
    ],
)

py_test(
    name = "test_weight_only_int8",
    srcs = ["test_weight_only_int8.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:safetensors"],
)

py_test(
    name = "test_weight_only_int8_sm8x",
    srcs = ["test_weight_only_int8.py", "test_weight_only_int8_sm8x.py"],
    main = "test_weight_only_int8_sm8x.py",
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors"],
)

[
    py_test(
        name = "test_weight_only_int8_" + name,
        srcs = ["test_weight_only_int8.py", "test_weight_only_int8_sm90_sm100.py"],
        main = "test_weight_only_int8_sm90_sm100.py",
        env = {"W8A16_EXPECT_SM": major},
        tags = [gpu],
        exec_properties = {"gpu": gpu, "gpu_count": "1"},
        deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors"],
    )
    for name, major, gpu in [
        ("sm90", "9", "H20_CU13"),
        ("sm100", "10", "SM100_ARM_CU13"),
    ]
]

py_test(
    name = "test_groupwise_checkpoint",
    srcs = ["test_groupwise_checkpoint.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm/models_py:groupwise_export_helpers",
        "//rtp_llm:safetensors_torch",
    ],
)

py_test(
    name = "test_groupwise_owners",
    srcs = ["test_groupwise_owners.py", "test_groupwise_checkpoint.py", "staged_checkpoint_fixture.py"],
    main = "test_groupwise_owners.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors_torch"],
)

py_test(
    name = "test_block_fp8_ft",
    srcs = ["test_block_fp8_ft.py"],
    env = checkpoint_reader_test_env({}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_glm_ft_block_tree",
    srcs = ["test_glm_ft_block_tree.py", "test_block_fp8_ft.py"],
    main = "test_glm_ft_block_tree.py",
    env = checkpoint_reader_test_env({}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_glm_ft_block_serving",
    srcs = ["test_glm_ft_block_serving.py", "test_glm_ft_block_tree.py", "test_block_fp8_ft.py"],
    main = "test_glm_ft_block_serving.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    env = checkpoint_reader_test_env({}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm:config",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/tools/convert:convert",
    ],
    exec_properties = select({
        "//:using_cuda13_x86": {"gpu": "H20_CU13"},
        "//conditions:default": {"gpu": "H20"},
    }),
    tags = ["H20", "H20_CU13"],
)

py_test(
    name = "test_moe_ft_scalar",
    srcs = ["test_moe_ft_scalar.py", "test_block_fp8_ft.py"],
    main = "test_moe_ft_scalar.py",
    env = checkpoint_reader_test_env({}),
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "test_w8a8_int8",
    env = checkpoint_reader_test_env({}),
    srcs = ["test_w8a8_int8.py"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch", "//rtp_llm:safetensors"],
)

py_test(
    name = "qwen_quant_dialects_test",
    srcs = ["qwen_quant_dialects_test.py"],
    env = checkpoint_reader_test_env(),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors_torch"],
)

py_test(
    name = "test_smooth_omni",
    srcs = ["test_smooth_omni.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch", "//rtp_llm:safetensors"],
)

py_test(
    name = "test_smooth_omni_h20",
    srcs = ["test_smooth_omni.py"],
    main = "test_smooth_omni.py",
    env = {"TEST_SMOOTH_OMNI_CUDA": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch", "//rtp_llm:safetensors"],
)

py_test(
    name = "test_w8a8_int8_moe",
    env = checkpoint_reader_test_env({}),
    srcs = ["test_w8a8_int8_moe.py"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch", "//rtp_llm:safetensors"],
)

py_test(
    name = "test_load_time_w8a8",
    srcs = ["test_load_time_w8a8.py", "test_w8a8_int8_moe.py"],
    main = "test_load_time_w8a8.py",
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors", "//rtp_llm:w8a8_int8_quant"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
)

py_test(
    name = "test_w4a8_checkpoint",
    env = checkpoint_reader_test_env({}),
    srcs = ["test_w4a8_checkpoint.py", "test_w8a8_int8_moe.py", "staged_checkpoint_fixture.py"],
    main = "test_w4a8_checkpoint.py",
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch", "//rtp_llm:safetensors"],
)

py_test(
    name = "groupwise_runtime_test",
    srcs = ["groupwise_runtime_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

# Native execution is opt-in after an idle ROCm GPU has been allocated.
py_test(
    name = "groupwise_runtime_h20_test",
    srcs = ["groupwise_runtime_test.py"],
    main = "groupwise_runtime_test.py",
    env = {"RUN_GROUPWISE_NATIVE_GPU_TESTS": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch"],
)

py_test(
    name = "groupwise_runtime_rocm_test",
    srcs = ["groupwise_runtime_test.py"],
    main = "groupwise_runtime_test.py",
    env = {"RUN_GROUPWISE_NATIVE_GPU_TESTS": "1", "GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    target_compatible_with = select({
        "//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)
