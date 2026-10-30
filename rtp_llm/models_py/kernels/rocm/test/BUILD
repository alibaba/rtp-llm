py_test(
    name = "test_lm_head_bf16x3",
    srcs = ["test_lm_head_bf16x3.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:numpy",
        "//rtp_llm:triton",
        "//rtp_llm:aiter",
    ],
    timeout = "long",
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
