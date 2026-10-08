py_test(
    name = "linear_replay_test",
    srcs = ["linear_replay_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)
