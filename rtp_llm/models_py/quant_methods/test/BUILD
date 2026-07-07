py_test(
    name = "test_fp8_already_quantized",
    srcs = ["test_fp8_already_quantized.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
)

py_test(
    name = "verify_phase1_dispatch",
    srcs = ["verify_phase1_dispatch.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_load_integrity",
    srcs = ["test_load_integrity.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
