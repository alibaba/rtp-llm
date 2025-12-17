test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fp8_deepgemm_linear_test",
    srcs = ["fp8_deepgemm_linear_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    deps = py_test_deps,
    env = dict(
        DEVICE_RESERVE_MEMORY_BYTES = "512000000",
        FLASHINFER_DISABLE_VERSION_CHECK = "1",
        CUDA_VISIBLE_DEVICES = "2",
    ),
    tags = ["L20A"],
    exec_properties = {'gpu':'L20A'},
)

