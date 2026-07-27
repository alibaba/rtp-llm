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

py_test(
    name = "collective_torch_hipgraph_unit_test",
    srcs = ["collective_torch_hipgraph_unit_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
)

py_test(
    name = "collective_lifecycle_test",
    srcs = ["collective_lifecycle_test.py"],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
)

py_test(
    name = "deepep_lifecycle_test",
    srcs = ["deepep_lifecycle_test.py"],
    data = [
        "//rtp_llm/models_py:modules",
    ],
    deps = [
        "//rtp_llm/models_py/distributed:deepep_wrapper",
    ],
    timeout = "short",
)

py_test(
    name = "symm_mem_sleep_level_test",
    srcs = ["symm_mem_sleep_level_test.py"],
    deps = [
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
)

py_test(
    name = "mega_symm_lifecycle_test",
    srcs = ["mega_symm_lifecycle_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    timeout = "short",
)

py_test(
    name = "checkpoint_symm_mem_integration_test",
    srcs = ["checkpoint_symm_mem_integration_test.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:utils",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    size = "large",
    timeout = "long",
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "local", "no-remote", "H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "2"},
)

py_test(
   name = "user_buffers_test",
   srcs = ["user_buffers_test.py"],
   deps = [
       "//rtp_llm:testlib",
       "//rtp_llm/models_py/distributed:user_buffers",
       "//rtp_llm/models_py/distributed:collective_torch"
   ],
   env = {"GPU_COUNT": "4"},
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20', 'gpu_count':'4'},
)
