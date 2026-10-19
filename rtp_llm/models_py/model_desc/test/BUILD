py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]


py_test(
    name = "diffusion_gemma_moe_ab_test",
    srcs = [
        "diffusion_gemma_moe_ab_test.py",
        "diffusion_gemma_realweight_test.py",
        "diffusion_gemma_test.py",
    ],
    main = "diffusion_gemma_moe_ab_test.py",
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
    env = {
        "DGM_REAL_CKPT": "/mnt/fuse/smoke-test/hf/google--diffusiongemma-26B-A4B-it--f7f5b7f5fa82ffc52addd066915886d497f5517b",
    },
    # Fused-vs-loop expert A/B with real weights (bf16-noise guard); needs
    # the FUSE-staged checkpoint, so it is a manual remote/CI GPU target.
    tags = ["H20_CU13"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    timeout = "eternal",
)

py_test(
    name = "diffusion_gemma_reuse_ab_test",
    srcs = [
        "diffusion_gemma_realweight_test.py",
        "diffusion_gemma_test.py",
    ],
    main = "diffusion_gemma_realweight_test.py",
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
    ],
    env = {
        "DGM_REAL_CKPT": "/mnt/fuse/smoke-test/hf/google--diffusiongemma-26B-A4B-it--f7f5b7f5fa82ffc52addd066915886d497f5517b",
    },
    tags = ["H20_CU13"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    timeout = "eternal",
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
    name = "diffusion_gemma_test",
    srcs = ["diffusion_gemma_test.py"],
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
    ],
    size = "small",
    # Imports rtp_llm.ops (librtp_compute_ops links libcuda): needs a GPU
    # worker. Untagged, the UT lanes schedule it on CPU-only workers where
    # the import dies with libcuda.so.1 not found (r61 ut-sm8x).
    tags = [
        "H20_CU13",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "diffusion_gemma_realweight_test",
    srcs = [
        "diffusion_gemma_realweight_test.py",
        "diffusion_gemma_test.py",
    ],
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
    ],
    size = "medium",
    # Real-weights layer math check; runs only with DGM_REAL_CKPT set
    # (skipped in CI). Use --spawn_strategy=standalone locally so the
    # host-visible checkpoint path resolves.
    # Imports rtp_llm.ops (librtp_compute_ops links libcuda): needs a GPU
    # worker. Untagged, the UT lanes schedule it on CPU-only workers where
    # the import dies with libcuda.so.1 not found (r61 ut-sm8x).
    tags = [
        "H20_CU13",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)


py_test(
    name = "diffusion_gemma_loader_test",
    srcs = ["diffusion_gemma_loader_test.py"],
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
    ],
    size = "small",
)

py_test(
    name = "diffusion_gemma_proposer_test",
    srcs = [
        "diffusion_gemma_proposer_test.py",
        "diffusion_gemma_test.py",
    ],
    main = "diffusion_gemma_proposer_test.py",
    data = ["//:rtp_compute_ops"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:lora",
        "//rtp_llm:models",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib_lite",
        "//rtp_llm/models_py:models",
    ],
    size = "small",
    # Imports rtp_llm.ops (librtp_compute_ops links libcuda): needs a GPU
    # worker. Untagged, the UT lanes schedule it on CPU-only workers where
    # the import dies with libcuda.so.1 not found (r61 ut-sm8x).
    tags = [
        "H20_CU13",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)

py_test(
    name = "hidden_state_capture_test",
    srcs = ["hidden_state_capture_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
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
