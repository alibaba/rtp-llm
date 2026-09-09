
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_dpsk_bf16",
    srcs = ["test_dpsk_bf16.py", "util.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)

# test_dpsk32_fp8 / test_model1_fp8 verified on sm86: 3/3 and 5/5 pass, 0 skips.
# The fp8_ds_mla write path uses __nv_cvt_float_to_fp8, which has a software
# conversion below sm89 and is compiled for 8.6 by our AOT arch list.
py_test(
    name = "test_dpsk32_fp8",
    srcs = ["test_dpsk32_fp8.py", "util.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_model1_fp8",
    srcs = ["test_model1_fp8.py", "util.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)