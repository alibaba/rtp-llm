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
)

py_test(
    name = "test_qwen3_dspark",
    srcs = ["test_qwen3_dspark.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
