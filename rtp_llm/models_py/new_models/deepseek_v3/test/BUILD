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
