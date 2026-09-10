py_test(
    name = "zigzag_permute_test",
    srcs = ["zigzag_permute_test.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:triton",
        "//rtp_llm/models_py/triton_kernels:zigzag_permute",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "B300", "gpu_count": "1"},
    tags = ["B300"],
    timeout = "short",
)

py_test(
    name = "linear_replay_test",
    srcs = ["linear_replay_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:linear_replay",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)
