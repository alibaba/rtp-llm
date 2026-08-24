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

# Library for attention reference implementation
py_library(
    name = "atten_test_util",
    srcs = ["atten_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "trtllm_gen_test",
    srcs = ["trtllm_gen_test.py"],
    deps = [
        ":atten_test_util",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'}
)

flashinfer_test_deps = py_test_deps + select({
    "@//:using_cuda12_9_x86": flashinfer_deps,
    "@//:cuda_pre_12_9": flashinfer_deps,
    "//conditions:default": [],
})

flashinfer_platforms = [
    {"sm": "sm90", "gpu": "H20"},
    {"sm": "sm8x", "gpu": "L20"},
    {"sm": "sm12x", "gpu": "RTX_5000_PRO"},
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
        main = test["main"],
        srcs = [test["main"]] + test["extra_srcs"],
        deps = flashinfer_test_deps,
        tags = [platform["gpu"]],
        exec_properties = {
            "gpu": platform["gpu"],
        },
    )
    for test in flashinfer_attention_tests
    for platform in flashinfer_platforms
]

py_test(
    name = "test_xqa",
    srcs = [
        "test_xqa.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": ["//rtp_llm:rtp-kernel"],
        "//conditions:default": [],
    }),
    tags = ["H20", "cuda12_9"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {
        'gpu': 'H20',
    },
)

# FlashInfer TRT-LLM FMHA v2 tests
py_test(
    name = "test_trtllm_fmha_v2_paged_prefill_sm90",
    main = "test_trtllm_fmha_v2_paged_prefill.py",
    srcs = [
        "test_trtllm_fmha_v2_paged_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20", "cuda12_9"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trtllm_fmha_v2_paged_prefill_sm12x",
    main = "test_trtllm_fmha_v2_paged_prefill.py",
    srcs = [
        "test_trtllm_fmha_v2_paged_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["RTX_5000_PRO"],
    exec_properties = {
        'gpu': 'RTX_5000_PRO',
    },
)

py_test(
    name = "test_trtllm_fmha_v2_prefill_sm90",
    main = "test_trtllm_fmha_v2_prefill.py",
    srcs = [
        "test_trtllm_fmha_v2_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20", "cuda12_9"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trtllm_fmha_v2_prefill_sm12x",
    main = "test_trtllm_fmha_v2_prefill.py",
    srcs = [
        "test_trtllm_fmha_v2_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + [
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["RTX_5000_PRO"],
    exec_properties = {
        'gpu': 'RTX_5000_PRO',
    },
)

# Benchmark for FlashInfer tensor cores performance comparison
py_test(
    name = "bench_flashinfer_tensor_cores",
    srcs = [
        "bench_flashinfer_tensor_cores.py",
        "bench_utils.py",
    ],
    deps = py_test_deps,
    tags = ["manual", "H20"],  # manual tag for benchmark tests
    exec_properties = {
        'gpu': 'H20',
    },
)

# Test for PyFlashinferPrefillPagedAttnOp CUDA graph path vs normal path
py_test(
    name = "test_py_flashinfer_paged_prefill_cuda_graph",
    srcs = [
        "test_flashinfer_prefill/test_py_flashinfer_paged_prefill_cuda_graph.py",
        "base_attention_test.py"
    ],
    deps = flashinfer_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Test for MhaRotaryEmbeddingOp - RoPE implementation for MHA
py_test(
    name = "test_mha_rotary_emb",
    srcs = [
        "test_flashinfer_prefill/test_mha_rotary_emb.py",
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# PrefillImpl BenchTest
# Manual developer benchmark; correctness is covered by the test matrix above.
py_test(
    name = "bench_prefill_impls",
    main = "bench/bench_prefill_impls.py",
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
    deps = py_test_deps + [
        ":filelock",
        "//rtp_llm/test/utils:test_util",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps + ["//rtp_llm:rtp-kernel"],
        "//conditions:default": [],
    }),
    tags = ["manual", "H20", "cuda12_9"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {
        'gpu': 'H20',
        'gpu_count': '1',
    },
)
