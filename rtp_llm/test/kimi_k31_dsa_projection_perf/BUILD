py_test(
    name = "dsa_projection_benchmark",
    srcs = ["dsa_projection_benchmark.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["manual", "SM103_CU13"],
    timeout = "long",
)
