py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "mxfp8_cold_transition_test",
    srcs = ["mxfp8_cold_transition_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "fp8_deepgemm_linear_test",
    srcs = ["fp8_deepgemm_linear_test.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fp8_deepgemm_linear_out_test",
    srcs = ["fp8_deepgemm_linear_out_test.py"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_sm100_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_test.py",
        "fp8_deepgemm_linear_test.py"
    ],
    deps = py_test_deps,
    tags = ["SM100"],
    exec_properties = {'gpu':'SM100'},
)

py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_deepgemm_linear_test.py"
    ],
    deps = py_test_deps,
    tags = ["SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM'},
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    deps = py_test_deps,
    tags = ["SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM'},
)


py_test(
    name = "fp8_deepgemm_linear_external_scale_test",
    srcs = ["fp8_deepgemm_linear_external_scale_test.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "mxfp8_linear_test",
    srcs = ["mxfp8_linear_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {'gpu': "SM103_CU13", 'gpu_count': '1'},
)

py_test(
    name = "mxfp8_prefill_quant_test",
    srcs = ["mxfp8_prefill_quant_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {'gpu': "SM103_CU13", 'gpu_count': '1'},
)

py_test(
    name = "mxfp8_launch_test",
    srcs = ["mxfp8_launch_test.py", "mxfp8_prefill_quant_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {'gpu': "SM103_CU13", 'gpu_count': '1'},
)

py_test(
    name = "mxfp8_dispatch_test",
    srcs = ["mxfp8_dispatch_test.py", "mxfp8_prefill_quant_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {'gpu': "SM103_CU13", 'gpu_count': '1'},
)

py_test(
    name = "mxfp8_workspace_test",
    srcs = ["mxfp8_workspace_test.py"],
    deps = py_test_deps + ["//rtp_llm:nvidia-cutlass-dsl"],
    tags = ["SM103_CU13"],
    exec_properties = {'gpu': "SM103_CU13", 'gpu_count': '1'},
)
