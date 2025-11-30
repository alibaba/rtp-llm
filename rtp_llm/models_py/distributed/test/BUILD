py_test (
    name = "deepep_test",
    srcs = ["deepep_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test(
    name = "collective_torch_test",
    srcs = ["collective_torch_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = 'moderate',
    env = {"GPU_COUNT": "4"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'4'},
)
