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
