py_test(
    name = "forward_expert_stats_test",
    srcs = ["test_forward_expert_stats.py"],
    main = "test_forward_expert_stats.py",
    deps = [
        "//rtp_llm/models_py:utils",
        "//rtp_llm:torch",
    ],
    env = {"GPU_COUNT": "0"},
    tags = ["manual"],
)
