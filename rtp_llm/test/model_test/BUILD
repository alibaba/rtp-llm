py_test(
    name = "test_mimo_v25",
    srcs = ["test_mimo_v25.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ],
    # Needs a real MiMo-V2.5 checkpoint (CHECKPOINT_PATH) and 4 GPUs.
    tags = ["manual"],
    timeout = "eternal",
)

py_test(
    name = "benchmark_mimo_v25_gsm8k",
    srcs = ["benchmark_mimo_v25_gsm8k.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
    ],
    deps = [
        "//rtp_llm:requests",
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ],
    # Manual accuracy benchmark: loads the real checkpoint on four GPUs and
    # evaluates 200 GSM8K prompts by default.
    tags = ["manual"],
    timeout = "eternal",
)
