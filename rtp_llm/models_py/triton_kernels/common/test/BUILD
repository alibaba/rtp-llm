py_test (
    name = "silu_mul_masked_test",
    srcs = ["silu_mul_masked_test.py"],
    deps = [
        "//rtp_llm:matplotlib",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test(
    name = "test_moe_gating",
    srcs = ["test_moe_gating.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "test_fused_add_rmsnorm_fp8_quant",
    srcs = ["test_fused_add_rmsnorm_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "test_fused_rmsnorm_gated_fp8_quant",
    srcs = ["test_fused_rmsnorm_gated_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "test_silu_and_mul_dense_fp8_quant",
    srcs = ["test_silu_and_mul_dense_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "test_attn_output_gate",
    srcs = ["test_attn_output_gate.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

# Perf target driven by skill rtp-kernel-benchmark.
# Run via: scripts/run_kernel_benchmark.sh ... --target //rtp_llm/models_py/triton_kernels/common/test:bench_all_fuse_kernels_perf
# Tagged "manual" so default `bazel test //...` will not pick it up.
py_test(
    name = "bench_all_fuse_kernels_perf",
    srcs = ["bench_all_fuse_kernels.py"],
    main = "bench_all_fuse_kernels.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20", "manual"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

# Benchmark: MXFP8 SwiGLU-OAI fused kernel vs unfused path (MiniMax-M3 shared expert).
# Run: bazelisk test --config=cuda13 --config=sm10x \
#     //rtp_llm/models_py/triton_kernels/common/test:bench_mxfp8_swiglu_oai_fused_vs_unfused
py_test(
    name = "bench_mxfp8_swiglu_oai_fused_vs_unfused",
    srcs = ["bench_mxfp8_swiglu_oai_fused_vs_unfused.py"],
    main = "bench_mxfp8_swiglu_oai_fused_vs_unfused.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {
        "GPU_COUNT": "1",
        "DG_JIT_CACHE_DIR": "/tmp/deep_gemm_jit_bench",
    },
    tags = ["manual"],
)
