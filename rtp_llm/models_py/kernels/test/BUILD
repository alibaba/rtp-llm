py_test (
    name = "deepgemm_test",
    srcs = ["deepgemm_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:kernels",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)
