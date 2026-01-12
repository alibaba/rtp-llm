
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "per_token_group_quant_8bit_test",
    srcs = ["per_token_group_quant_8bit_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_fp8_grouped_gemm_test",
    srcs = ["cutlass_fp8_grouped_gemm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_tensor_scaled_fp8_quant_test",
    srcs = ["per_tensor_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_token_scaled_fp8_quant_test",
    srcs = ["per_token_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_fp4_scaled_mm_test",
    srcs = ["cutlass_fp4_scaled_mm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["manual"],
)

py_test (
    name = "pack_ue8m0_kernel_test",
    srcs = ["pack_ue8m0_kernel_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_xqa_batch_decode",
    srcs = ["test_xqa_batch_decode.py"],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": ["//rtp_llm:flashinfer-python"],
        "@//:cuda_pre_12_9": ["//rtp_llm:flashinfer-python"],
        "//conditions:default": []
    }),
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

