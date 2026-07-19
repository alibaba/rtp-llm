py_test(
    name = "test_unquantized",
    srcs = ["test_unquantized.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_fp8_already_quantized",
    srcs = ["test_fp8_already_quantized.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_fp8_rocm_static",
    srcs = ["test_fp8_rocm_static.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
    ],
    tags = ["rocm"],
    exec_properties = {
        "gpu": "MI308X-ROCM7",
    },
)
