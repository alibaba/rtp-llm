py_test(
    name = "legacy_fp8_indexer_score_test",
    srcs = ["legacy_fp8_indexer_score_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    tags = ["SM100_ARM_CU13"],
)
py_test(
    name = "linear_replay_test",
    srcs = ["linear_replay_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    visibility = ["//visibility:public"],
)
