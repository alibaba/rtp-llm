py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_library(
    name = "sm120_test_utils",
    srcs = ["sm120_test_utils.py"],
    deps = [
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/test/utils:test_util",
    ],
)

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

# cuda12_9 x86-only contract: this target must import the SM120 binding on H20
# and prove that the host guard rejects launch before entering device code.
py_test(
    name = "fp8_vllm_blockwise_non_sm120_guard_test",
    srcs = [
        "fp8_vllm_blockwise_non_sm120_guard_test.py",
    ],
    deps = py_test_deps + [":sm120_test_utils"],
    tags = ["H20", "cuda12_9"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fp8_deepgemm_linear_quantized_input_contract_test",
    srcs = ["fp8_deepgemm_linear_quantized_input_contract_test.py"],
    deps = py_test_deps,
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

py_test(
    name = "fp8_vllm_blockwise_sm120_factory_test",
    srcs = ["fp8_vllm_blockwise_sm120_factory_test.py"],
    deps = py_test_deps,
    tags = ["H20", "cuda12_9"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fp8_vllm_blockwise_sm120_linear_test",
    srcs = [
        "fp8_vllm_blockwise_sm120_linear_test.py",
    ],
    deps = py_test_deps + [
        ":sm120_test_utils",
        "//rtp_llm/test/utils:test_util",
    ],
    tags = ["RTX_5000_PRO", "cuda12_9"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {'gpu':'RTX_5000_PRO'},
)

# Keep execution properties literal: the repository's py_test prelude inspects
# this dictionary before passing it to native.py_test, so select is unsupported.
py_test(
    name = "fp8_vllm_blockwise_sm120_linear_cu13_test",
    srcs = ["fp8_vllm_blockwise_sm120_linear_test.py"],
    main = "fp8_vllm_blockwise_sm120_linear_test.py",
    deps = py_test_deps + [
        ":sm120_test_utils",
        "//rtp_llm/test/utils:test_util",
    ],
    tags = ["RTX_5000_PRO_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {'gpu':'RTX_5000_PRO_CU13'},
)
