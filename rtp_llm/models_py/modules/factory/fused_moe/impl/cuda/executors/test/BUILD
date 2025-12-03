py_test (
    name = "deepgemm_masked_executor_test",
    srcs = ["deepgemm_masked_executor_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)