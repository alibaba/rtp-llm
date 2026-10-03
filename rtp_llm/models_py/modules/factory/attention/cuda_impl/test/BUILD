# Test configuration for PyFlashinferDecodeAttnOp

load("@arch_config//:arch_select.bzl", "requirement")

requirement(["filelock"])

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

CUDA13_X86_ONLY = select({
    "@//:using_cuda13_x86": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

CUDA13_ARM_ONLY = select({
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

# Library for attention reference implementation
py_library(
    name = "atten_test_util",
    srcs = ["atten_test_util.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "trtllm_gen_test",
    srcs = ["trtllm_gen_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "SM100_ARM_CU13",
        "gpu_count": "1",
    },
    tags = [
        "SM100_ARM_CU13",
        "open_skip",
    ],
    target_compatible_with = CUDA13_ARM_ONLY,
    deps = [
        ":atten_test_util",
        "//rtp_llm:testlib",
    ],
)

flashinfer_test_deps = py_test_deps + select({
    "@//:using_cuda13_x86": flashinfer_deps,
    "//conditions:default": [],
})

flashinfer_platforms = [
    {
        "sm": "sm90",
        "gpu": "H20_CU13",
    },
    {
        "sm": "sm8x",
        "gpu": "L20_CU13",
    },
    {
        "sm": "sm12x",
        "gpu": "RTX_5000_PRO_CU13",
    },
]

flashinfer_attention_tests = [
    {
        "name": "test_py_flashinfer_mha_decode",
        "main": "test_py_flashinfer_mha_decode.py",
        "extra_srcs": [
            "attention_ref.py",
            "base_attention_test.py",
        ],
    },
    {
        "name": "test_py_flashinfer_ragged_mha_prefill",
        "main": "test_flashinfer_prefill/test_py_flashinfer_ragged_mha_prefill.py",
        "extra_srcs": [
            "attention_ref.py",
            "base_attention_test.py",
        ],
    },
    {
        "name": "test_py_flashinfer_paged_mha_prefill",
        "main": "test_flashinfer_prefill/test_py_flashinfer_paged_mha_prefill.py",
        "extra_srcs": [
            "attention_ref.py",
            "base_attention_test.py",
        ],
    },
    {
        "name": "test_py_flashinfer_hybrid_mha_prefill",
        "main": "test_flashinfer_prefill/test_py_flashinfer_hybrid_mha_prefill.py",
        "extra_srcs": [
            "base_attention_test.py",
        ],
    },
]

# Generates one py_test per (test, platform): <test name>_<sm>,
# e.g. test_py_flashinfer_hybrid_mha_prefill_sm90.
[
    py_test(
        name = test["name"] + "_" + platform["sm"],
        srcs = [test["main"]] + test["extra_srcs"],
        exec_properties = {
            "gpu": platform["gpu"],
        },
        main = test["main"],
        tags = [platform["gpu"]],
        target_compatible_with = CUDA13_X86_ONLY,
        deps = flashinfer_test_deps,
    )
    for test in flashinfer_attention_tests
    for platform in flashinfer_platforms
]

py_test(
    name = "test_xqa",
    srcs = [
        "attention_ref.py",
        "base_attention_test.py",
        "test_xqa.py",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    tags = [
        "H20_CU13",
        "cuda13",
    ],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": ["//rtp_llm:rtp-kernel"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_flash_attn_3_graph_geometry",
    srcs = ["test_flash_attn_3.py"],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    main = "test_flash_attn_3.py",
    tags = ["H20_CU13"],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps,
)

# FlashInfer TRT-LLM FMHA v2 tests
py_test(
    name = "test_trtllm_fmha_v2_paged_prefill_sm90",
    srcs = [
        "attention_ref.py",
        "base_attention_test.py",
        "test_trtllm_fmha_v2_paged_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    main = "test_trtllm_fmha_v2_paged_prefill.py",
    tags = [
        "H20_CU13",
        "cuda13",
    ],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_trtllm_fmha_v2_paged_prefill_sm12x",
    srcs = [
        "attention_ref.py",
        "base_attention_test.py",
        "test_trtllm_fmha_v2_paged_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
    ],
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
    },
    main = "test_trtllm_fmha_v2_paged_prefill.py",
    tags = ["RTX_5000_PRO_CU13"],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_trtllm_fmha_v2_prefill_sm90",
    srcs = [
        "attention_ref.py",
        "base_attention_test.py",
        "test_trtllm_fmha_v2_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    main = "test_trtllm_fmha_v2_prefill.py",
    tags = [
        "H20_CU13",
        "cuda13",
    ],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_trtllm_fmha_v2_prefill_sm12x",
    srcs = [
        "attention_ref.py",
        "base_attention_test.py",
        "test_trtllm_fmha_v2_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
    ],
    exec_properties = {
        "gpu": "RTX_5000_PRO_CU13",
    },
    main = "test_trtllm_fmha_v2_prefill.py",
    tags = ["RTX_5000_PRO_CU13"],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": flashinfer_deps,
        "//conditions:default": [],
    }),
)

# Benchmark for FlashInfer tensor cores performance comparison
py_test(
    name = "bench_flashinfer_tensor_cores",
    srcs = [
        "bench_flashinfer_tensor_cores.py",
        "bench_utils.py",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    tags = [
        "H20_CU13",
        "manual",
    ],  # manual tag for benchmark tests
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps,
)

# Test for PyFlashinferPrefillPagedAttnOp CUDA graph path vs normal path
# Must stay on H20: verified on sm86 as 23 tests / 22 pass / 1 fail
# (test_q_cast_buffer_reused_across_replay), because attn_q_dtype gates the FA3
# FP8-Q path on is_sm90() (py_flashinfer_mha.py:133-139). This is the only
# coverage of that path.
py_test(
    name = "test_py_flashinfer_paged_prefill_cuda_graph",
    srcs = [
        "base_attention_test.py",
        "test_flashinfer_prefill/test_py_flashinfer_paged_prefill_cuda_graph.py",
    ],
    exec_properties = {
        "gpu": "H20_CU13",
    },
    tags = ["H20_CU13"],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = flashinfer_test_deps,
)

# Test for MhaRotaryEmbeddingOp - RoPE implementation for MHA
# Verified on sm86: 1 test, 1 pass, 0 skips. RoPE is arch-agnostic and the
# FlashInfer fa2 fallback works, so this runs on the default A10 worker.
py_test(
    name = "test_mha_rotary_emb",
    srcs = [
        "test_flashinfer_prefill/test_mha_rotary_emb.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = flashinfer_test_deps,
)

# PrefillImpl BenchTest
# Manual developer benchmark; correctness is covered by the test matrix above.
py_test(
    name = "bench_prefill_impls",
    srcs = [
        "base_attention_test.py",
        "bench/bench_prefill_impls.py",
        "bench/core.py",
        "bench/prefill_benches.py",
        "bench/scheduler.py",
        "bench_utils.py",
    ],
    args = [
        "--q-dtype=bf16",
        "--kv-cache-dtype=bf16,fp8",
        "--input-len=4096,8192,16384,65536,131072",
        "--reuse-cache-ratio=0,0.3,0.7,0.9",
        "--batch-size=1",
        "--preset=qwen3-8b",
        "--page-size=64",
        "--repeat=15",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    main = "bench/bench_prefill_impls.py",
    tags = [
        "H20_CU13",
        "cuda13",
        "manual",
    ],
    target_compatible_with = CUDA13_X86_ONLY,
    deps = py_test_deps + [
        ":filelock",
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda13_x86": flashinfer_deps + ["//rtp_llm:rtp-kernel"],
        "//conditions:default": [],
    }),
)
