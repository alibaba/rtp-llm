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

py_test (
    name = "moriep_test",
    srcs = ["moriep_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'8'},
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
    tags = ["rocm"],
)

py_test(
    name = "collective_torch_allreduce_tier_test",
    srcs = ["collective_torch_allreduce_tier_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
    tags = ["rocm"],
)

py_test(
    name = "rocm_allreduce_tier_integration_test",
    srcs = ["rocm_allreduce_tier_integration_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py:modules",
    ],
    timeout = "moderate",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
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

py_test(
    name = "cp_moe_allgather_rs_test",
    srcs = ["cp_moe_allgather_rs_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = 'moderate',
    env = {"GPU_COUNT": "4"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'4'},
)
