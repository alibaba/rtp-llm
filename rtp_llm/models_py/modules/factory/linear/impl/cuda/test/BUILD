py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fp8_deepgemm_linear_out_test",
    srcs = ["fp8_deepgemm_linear_out_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
    # Imports the real kernel chain (rtp_llm.ops.compute_ops ->
    # librtp_compute_ops), so it needs the full testlib and a GPU pool.
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "fp8_linear_test",
    srcs = ["fp8_linear_test.py"],
    deps = py_test_deps,
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "fp8_deepgemm_linear_sm100_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_test.py",
        "fp8_linear_test.py",
        "fp8_deepgemm_linear_sm120_test.py",
    ],
    deps = py_test_deps,
    tags = ["SM100"],
    exec_properties = {'gpu':'SM100'},
)

py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    deps = py_test_deps,
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)

py_test(
    name = "fp8_deepgemm_linear_sm120_test",
    srcs = [
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    deps = py_test_deps,
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    deps = py_test_deps,
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)
