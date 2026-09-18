# Qwen3.5 decode fusion UTs. GPU-only; compare fused Triton vs the old path.

_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

_GPU_ENV = {"GPU_COUNT": "1"}

py_test(
    name = "test_env",
    srcs = ["test_env.py"],
    deps = _TEST_DEPS,
)

py_test(
    name = "test_fused_add_rmsnorm_fp8_quant",
    srcs = ["test_fused_add_rmsnorm_fp8_quant.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["L20"],
    exec_properties = {"gpu": "L20", "gpu_count": "1"},
)

py_test(
    name = "test_rmsnorm_gated_fp8_quant",
    srcs = ["test_rmsnorm_gated_fp8_quant.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["L20"],
    exec_properties = {"gpu": "L20", "gpu_count": "1"},
)

py_test(
    name = "test_conv1d_gdn_gating",
    srcs = ["test_conv1d_gdn_gating.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["L20"],
    exec_properties = {"gpu": "L20", "gpu_count": "1"},
)

py_test(
    name = "test_fused_qk_rmsnorm",
    srcs = ["test_fused_qk_rmsnorm.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["L20"],
    exec_properties = {"gpu": "L20", "gpu_count": "1"},
)

py_test(
    name = "test_sigmoid_mul_fp8_quant",
    srcs = ["test_sigmoid_mul_fp8_quant.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["L20"],
    exec_properties = {"gpu": "L20", "gpu_count": "1"},
)

py_test(
    name = "test_shared_inputs",
    srcs = ["test_shared_inputs.py"],
    deps = _TEST_DEPS,
    env = _GPU_ENV,
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_mega_shared_production",
    srcs = ["test_mega_shared_production.py"],
    deps = _TEST_DEPS,
    env = {"GPU_COUNT": "4"},
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "4"},
)
