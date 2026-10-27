py_test(
    name = "test_beam_search_cascade",
    srcs = ["test_beam_search_cascade.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:attention",
        "//rtp_llm:torch",
    ],
    timeout = "long",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
