package(default_visibility = ["//visibility:public"])

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# SM120 probe for the legacy NVFP4 CUTLASS skip contract.
py_test(
    name = "cutlass_fp4_scaled_mm_sm120_test",
    srcs = ["cutlass_fp4_scaled_mm_sm120_test.py"],
    deps = py_test_deps,
)
