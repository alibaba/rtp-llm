py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "qwen_gdn_graph_replay_test_rocm",
    srcs = ["qwen_gdn_graph_replay_test.py"],
    data = ["//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    main = "qwen_gdn_graph_replay_test.py",
    tags = ["rocm"],
    deps = py_test_deps,
)

py_test(
    name = "qwen_gdn_graph_replay_test",
    srcs = ["qwen_gdn_graph_replay_test.py"],
    data = ["//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    main = "qwen_gdn_graph_replay_test.py",
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "qwen3_next_moe_test",
    srcs = ["qwen3_next_moe_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = ["A10_CU13"],
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:models",
    ],
)

py_test(
    name = "generic_moe_allreduce_test",
    srcs = ["generic_moe_allreduce_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    # CPU-only contract test (mocks + CPU tensors); needs an accelerator image
    # solely to import librtp_compute_ops, so any CUDA worker works — no H20 pin.
    tags = ["A10_CU13"],
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm/models_py:models",
    ],
)

py_test(
    name = "generic_moe_allreduce_test_rocm",
    srcs = ["generic_moe_allreduce_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    main = "generic_moe_allreduce_test.py",
    tags = ["rocm"],
    deps = py_test_deps + [
        "//rtp_llm:config",
        "//rtp_llm/models_py:models",
    ],
)

py_test(
    name = "attention_input_routing_test",
    srcs = ["attention_input_routing_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    # Mock-based routing test on CPU tensors; needs only a generic CUDA worker
    # (librtp_compute_ops import), not an H20.
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    env = {"GPU_COUNT": "1"},
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
    tags = ["H20_CU13"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test_rocm",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    main = "qwen3_next_qkvz_ba_fusion_test.py",
    tags = ["rocm"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
