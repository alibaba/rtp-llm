py_library(
    name = "atten_test_util",
    srcs = ["atten_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "trtllm_gen_test",
    srcs = ["trtllm_gen_test.py"],
    deps = [
        ":atten_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "B200"],
    exec_properties = {'gpu':'B200', 'gpu_count':'1'},
)
