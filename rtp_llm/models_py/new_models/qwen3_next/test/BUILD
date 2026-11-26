load("//rtp_llm/models_py:checkpoint_test_platforms.bzl", "checkpoint_reader_test_env")

py_test(
    name = "load_time_w8a8_test",
    srcs = ["load_time_w8a8_test.py", "main_checkpoint_test.py"],
    main = "load_time_w8a8_test.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors_torch", "//rtp_llm:w8a8_int8_quant"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
)

py_test(
    name = "gated_fp8_checkpoint_test",
    srcs = ["gated_fp8_checkpoint_test.py", "main_checkpoint_test.py"],
    tags = ["H20_CU13"],
    env = checkpoint_reader_test_env({"GPU_COUNT": "2"}),
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "2"},
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors_torch"],
)

py_test(
    name = "scratch_policy_test",
    env = checkpoint_reader_test_env({}),
    srcs = ["scratch_policy_test.py", "main_checkpoint_test.py"],
    main = "scratch_policy_test.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "hybrid_ft_test",
    env = checkpoint_reader_test_env({}),
    srcs = ["hybrid_ft_test.py", "main_checkpoint_test.py"],
    main = "hybrid_ft_test.py",
    deps = [
        "//rtp_llm/models_py:checkpoint_reader_test_support",
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py/new_models/kimi_linear/test:hybrid_ft_fixture",
        "//rtp_llm/models_py/new_models/kimi_k25/test:hybrid_ft_fixture",
        "//rtp_llm/tools/convert:convert",
    ],
)

py_test(
    name = "gdn_parameters_test",
    srcs = ["gdn_parameters_test.py"],
    env = checkpoint_reader_test_env({}),
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "main_checkpoint_test",
    env = checkpoint_reader_test_env({}),
    srcs = ["main_checkpoint_test.py"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "native_service_test",
    env = checkpoint_reader_test_env({}),
    srcs = ["native_service_test.py", "main_checkpoint_test.py"],
    main = "native_service_test.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm:models", "//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "resident_restart_test",
    env = checkpoint_reader_test_env({}),
    srcs = ["resident_restart_test.py", "main_checkpoint_test.py"],
    main = "resident_restart_test.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
    deps = ["//rtp_llm/models_py:checkpoint_reader_test_support", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm/resident_weights", "//rtp_llm:safetensors_torch"],
)
