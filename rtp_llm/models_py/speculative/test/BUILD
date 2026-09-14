# DSV4 python tests target the CUDA13/Blackwell stack (deep_gemm >= 2.6, dsv4
# kernels, pytest closure). They are skipped as incompatible on other platforms
# instead of failing on pools that can never satisfy those dependencies.
_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_dspark_proposer",
    srcs = ["test_dspark_proposer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
    # Imports libcuda (the DSpark proposer kernel wrappers): pin to a GPU
    # test pool now that unpinned cuda13 actions default to the CPU worker
    # pool. The x86 branch takes the L20 pool; the ARM lane (which also
    # runs this target) keeps the SM100_ARM_CU13 default. Every branch
    # carries a real pool value — an empty-dict default is NOT a GPU
    # request and lands the test on a CPU worker.
    exec_properties = select({
        "@//:using_cuda13_x86": {"gpu": "L20_CU13", "gpu_count": "1"},
        "@//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    }),
)

py_test(
    name = "test_qwen3_dspark",
    srcs = ["test_qwen3_dspark.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_qwen3_dflash",
    srcs = ["test_qwen3_dflash.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dflash_weights",
    srcs = ["test_qwen3_dflash_weights.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
