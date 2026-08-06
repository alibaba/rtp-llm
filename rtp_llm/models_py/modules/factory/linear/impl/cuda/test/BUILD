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
    name = "fp8_linear_test",
    srcs = ["fp8_linear_test.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
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
    name = "fp8_deepgemm_linear_sm100_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_test.py",
        "fp8_linear_test.py"
    ],
    deps = py_test_deps,
    tags = ["SM100"],
    exec_properties = {'gpu':'SM100'},
)

py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_linear_test.py"
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
