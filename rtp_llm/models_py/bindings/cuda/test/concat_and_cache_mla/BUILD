py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_dpsk_bf16",
    srcs = [
        "test_dpsk_bf16.py",
        "util.py",
    ],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

# test_dpsk32_fp8 / test_model1_fp8: 3/3 and 5/5 pass, 0 skips on L20_CU13
# (sm89), re-confirmed after the sm8x lane moved to L20. The fp8_ds_mla write
# path uses __nv_cvt_float_to_fp8, which has a software conversion below sm89
# and a hardware one from sm89 up; the CUDA13 AOT arch list covers 8.9.
py_test(
    name = "test_dpsk32_fp8",
    srcs = [
        "test_dpsk32_fp8.py",
        "util.py",
    ],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_model1_fp8",
    srcs = [
        "test_model1_fp8.py",
        "util.py",
    ],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)