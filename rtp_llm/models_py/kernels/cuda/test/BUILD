
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "arm_wheel_import_test",
    srcs = ["arm_wheel_import_test.py"],
    deps = select({
        "@//:using_cuda12_arm": [
            "//rtp_llm:deep_ep",
            "//rtp_llm:deep_gemm",
            "//rtp_llm:fast-safetensors",
            "//rtp_llm:flash_attn",
            "//rtp_llm:flashinfer-python",
            "//rtp_llm:rtp-kernel",
        ],
        "//conditions:default": [],
    }),
    tags = ["SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM'},
)

py_test (
    name = "per_token_group_quant_8bit_test",
    srcs = ["per_token_group_quant_8bit_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_fp8_grouped_gemm_test",
    srcs = ["cutlass_fp8_grouped_gemm_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_w4a8_group_gemm_test",
    srcs = ["cutlass_w4a8_group_gemm_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_tensor_scaled_fp8_quant_test",
    srcs = ["per_tensor_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_token_scaled_fp8_quant_test",
    srcs = ["per_token_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_fp4_scaled_mm_test",
    srcs = ["cutlass_fp4_scaled_mm_test.py"],
    deps = py_test_deps,
    tags = ["manual"],
)

py_test (
    name = "pack_ue8m0_kernel_test",
    srcs = ["pack_ue8m0_kernel_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_xqa_batch_decode",
    srcs = ["test_xqa_batch_decode.py"],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": ["//rtp_llm:flashinfer-python"],
        "//conditions:default": []
    }),
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)
