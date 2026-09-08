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
    name = "pynccl_cp_unit_test",
    srcs = ["pynccl_cp_unit_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
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
    name = "tp_token_shard_test",
    srcs = ["tp_token_shard_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
)

py_test(
    name = "glm53_prefill_parallel_test",
    srcs = ["glm53_prefill_parallel_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/distributed:collective_torch"],
    timeout = "long",
)

py_test(
    name = "glm53_mhc_topk_test",
    srcs = ["glm53_mhc_topk_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm/models_py/distributed:collective_torch"],
    tags = ["L20A_DEV"],
    timeout = "long",
)

py_test(
    name = "glm53_collective_gemm_test",
    srcs = ["glm53_collective_gemm_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm/models_py/distributed:collective_torch"],
    tags = ["manual", "L20A_DEV"],
    timeout = "long",
)
