package(default_visibility = ["//visibility:public"])

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# SM120 strategy/construction cases covering the FP4 linear backend.
py_test(
    name = "fp4_gemm_linear_sm120_test",
    srcs = ["fp4_gemm_linear_sm120_test.py"],
    deps = py_test_deps,
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)
