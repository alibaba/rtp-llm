py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fp8_linear_test",
    srcs = ["fp8_linear_test.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fp8_deepgemm_linear_sm100_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_test.py",
        "fp8_linear_test.py"
    ],
    deps = py_test_deps,
    tags = ["SM100"],
    exec_properties = {'gpu':'SM100'},
)

# The Bazel config selects the CUDA-specific ARM pool for these shared targets.
py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_linear_test.py"
    ],
    deps = py_test_deps,
    tags = ["SM100_ARM"],
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    deps = py_test_deps,
    tags = ["SM100_ARM"],
)
