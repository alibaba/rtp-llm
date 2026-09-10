py_test (
    name = "silu_mul_masked_test",
    srcs = ["silu_mul_masked_test.py"],
    deps = [
        "//rtp_llm:matplotlib",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {
        "GPU_COUNT": "1",
    },
    # Not sm90-bound: the only hard constraint is Triton's fp8e4nv storage type,
    # which needs compute capability >= 89. A10 (sm86) would hard-fail with
    # "type fp8e4nv not supported in this architecture", so this runs on L20
    # (sm89) instead of H20. The H20 latency assertion in this file is inert in
    # CI (guarded by RUN_KERNEL_BENCHMARK and an "H20" device-name skip).
    tags = ["L20"],
    exec_properties = {'gpu':'L20', 'gpu_count':'1'},
)

py_test(
    name = "silu_mul_masked_heuristic_test",
    srcs = ["silu_mul_masked_heuristic_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_moe_gating",
    srcs = ["test_moe_gating.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {'gpu':'A10', 'gpu_count': '1'},
)

py_test(
    name = "scatter_qkv_test",
    srcs = ["scatter_qkv_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {'gpu':'A10', 'gpu_count': '1'},
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

