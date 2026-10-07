py_test(
    name = "test_native_q8_index_score",
    srcs = ["test_native_q8_index_score.py"],
    deps = [
        "//rtp_llm:fmha-sm100",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_native_q8_index_score_gpu",
    srcs = ["test_native_q8_index_score_gpu.py"],
    deps = [
        "//rtp_llm:fmha-sm100",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_nvfp4_q8_index_score",
    srcs = ["test_nvfp4_q8_index_score.py"],
    deps = [
        "//rtp_llm:fmha-sm100",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)
