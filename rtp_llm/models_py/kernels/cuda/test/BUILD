load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "per_token_group_quant_8bit_test",
    srcs = ["per_token_group_quant_8bit_test.py"],
    exec_properties = {"gpu": "L20_CU13"},
    # Not sm90-bound: the op under test has an sm86 cubin, but the in-test Triton
    # reference stores fp8e4nv, which Triton only allows at capability >= 89.
    # A10 (sm86) would hard-fail, so this runs on L20 (sm89) instead of H20.
    tags = [
        "L20_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "cutlass_fp8_grouped_gemm_test",
    srcs = ["cutlass_fp8_grouped_gemm_test.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "cutlass_w4a8_group_gemm_test",
    srcs = ["cutlass_w4a8_group_gemm_test.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "per_tensor_scaled_fp8_quant_test",
    srcs = ["per_tensor_scaled_fp8_quant_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    # Verified on sm86: bit-exact (torch.equal) vs the torch float8_e4m3fn
    # reference across 18 shapes, 0 skips. Our AOT cubins cover sm86.
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "per_token_scaled_fp8_quant_test",
    srcs = ["per_token_scaled_fp8_quant_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    # Verified on sm86: bit-exact (torch.equal) vs the torch float8_e4m3fn
    # reference across 18 shapes, 0 skips.
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "cutlass_fp4_scaled_mm_test",
    srcs = ["cutlass_fp4_scaled_mm_test.py"],
    tags = ["manual"],
    deps = py_test_deps,
)

py_test(
    name = "pack_ue8m0_kernel_test",
    srcs = ["pack_ue8m0_kernel_test.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps,
)

py_test(
    name = "fp8_quant_gemm_microbench",
    srcs = ["fp8_quant_gemm_microbench.py"],
    tags = ["manual"],
    deps = py_test_deps,
)

py_test(
    name = "test_xqa_batch_decode",
    srcs = ["test_xqa_batch_decode.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": ["//rtp_llm:flashinfer-python"],
        "@//:cuda_pre_12_9": ["//rtp_llm:flashinfer-python"],
        "//conditions:default": [],
    }),
)

cuda13_sm100_py_test(
    name = "fused_silu_mul_token_quant_batched_test",
    srcs = ["fused_silu_mul_token_quant_batched_test.py"],
    deps = py_test_deps,
)
