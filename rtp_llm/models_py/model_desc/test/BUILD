py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

k31_review_test_deps = py_test_deps + [
    "//rtp_llm:models",
    "//rtp_llm:testlib",
    "//rtp_llm/models_py:models",
]

py_test(
    name = "kimi_k31_review_regression_test",
    srcs = ["kimi_k31_review_regression_test.py"],
    deps = k31_review_test_deps,
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

py_test(
    name = "kimi_k31_norm_contract_test",
    srcs = ["kimi_k31_norm_contract_test.py"],
    deps = k31_review_test_deps,
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

py_test(
    name = "kimi_k31_device_input_test",
    srcs = ["kimi_k31_device_input_test.py"],
    deps = k31_review_test_deps,
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

py_test(
    name = "kimi_k31_capacity_contract_test",
    srcs = ["kimi_k31_capacity_contract_test.py"],
    deps = k31_review_test_deps,
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

py_test(
    name = "kimi_k31_cpu_weight_conversion_test",
    srcs = ["kimi_k31_cpu_weight_conversion_test.py"],
    deps = k31_review_test_deps,
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["SM103_CU13"],
)

py_test(
    name = "kimi_k31_prefill_layout_test",
    srcs = ["kimi_k31_prefill_layout_test.py"],
    deps = k31_review_test_deps,
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "2"},
    tags = ["L20_CU13", "multi_device"],
)

py_test(
    name = "qwen_gdn_graph_replay_test_rocm",
    srcs = ["qwen_gdn_graph_replay_test.py"],
    main = "qwen_gdn_graph_replay_test.py",
    data = ["//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
)

py_test(
    name = "qwen_gdn_graph_replay_test",
    srcs = ["qwen_gdn_graph_replay_test.py"],
    main = "qwen_gdn_graph_replay_test.py",
    data = ["//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
)

py_test(
    name = "qwen3_next_moe_test",
    srcs = ["qwen3_next_moe_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
)

py_test(
    name = "hidden_state_capture_test",
    srcs = ["hidden_state_capture_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "generic_moe_allreduce_test",
    srcs = ["generic_moe_allreduce_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    # CPU-only contract test (mocks + CPU tensors); needs an accelerator image
    # solely to import librtp_compute_ops, so any CUDA worker works — no H20 pin.
    tags = ["L20_CU13"],
)

py_test(
    name = "generic_moe_allreduce_test_rocm",
    srcs = ["generic_moe_allreduce_test.py"],
    main = "generic_moe_allreduce_test.py",
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
)

py_test(
    name = "attention_input_routing_test",
    srcs = ["attention_input_routing_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    # Mock-based routing test on CPU tensors; needs only a generic CUDA worker
    # (librtp_compute_ops import), not an H20.
    tags = ["L20_CU13"],
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    # Back on H20 after measuring on sm86: test_fused_slice_equals_separate_gemms
    # mismatches 282 of 4096 elements, greatest abs diff 0.25 against atol=1e-2.
    # Byte-identical across two sm86 runs, and it passes on H20. This is a plain
    # bf16 x @ w GEMM -- the tolerance was tuned for the sm90/hipBLASLt tiling
    # (see the test's own comment) and the sm86 cuBLAS kernel accumulates in a
    # different order. Staying on H20 by decision: loosening an exact-output check
    # to relocate a test trades real coverage for a scheduling preference.
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test_rocm",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    main = "qwen3_next_qkvz_ba_fusion_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
)

test_suite(
    name = "qwen4_exp_layer_test",
    tests = [
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:ple_cache_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:ple_graph_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:ple_host_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:ple_prefetch_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:qsa_wiring_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:wiring_test",
    ],
)

alias(
    name = "qwen4_exp_mtp_test",
    actual = "//rtp_llm/models_py/model_desc/test/qwen4_exp:mtp_test",
)

py_test(
    name = "pp_stage_construction_test",
    srcs = ["pp_stage_construction_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

py_test(
    name = "kimi_k3_fp8_attn_res_test",
    srcs = ["kimi_k3_fp8_attn_res_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:models",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "kimi_k3_mtp_local_projection_test",
    srcs = ["kimi_k3_mtp_local_projection_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:models",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "kimi_k3_multimodal_embedding_test",
    srcs = ["kimi_k3_multimodal_embedding_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py:models",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)
