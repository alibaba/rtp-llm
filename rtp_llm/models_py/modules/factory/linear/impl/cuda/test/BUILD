py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fp8_deepgemm_linear_out_test",
    srcs = ["fp8_deepgemm_linear_out_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    # Mocked contract test: CPU tensors only, fp8_gemm_nt and
    # sgl_per_token_group_quant_fp8 are patched out, so no DeepGEMM kernel runs.
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "fp8_linear_test",
    srcs = ["fp8_linear_test.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_quantized_input_contract_test",
    srcs = ["fp8_deepgemm_linear_quantized_input_contract_test.py"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_sm100_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_test.py",
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    exec_properties = {"gpu": "SM100"},
    tags = ["SM100"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
    tags = ["SM100_ARM_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_sm120_test",
    srcs = [
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
    tags = ["RTX_5000_PRO_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
    tags = ["SM100_ARM_CU13"],
    deps = py_test_deps,
)
