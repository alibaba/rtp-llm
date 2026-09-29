py_test(
    name = "vip_server_test",
    srcs = [
        "vip_server_test.py",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "frontend",
    ],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "frontend_start_test",
    srcs = [
        "frontend_app_test.py",
    ],
    env = {
        "CHECKPOINT_PATH": "/mnt/nas1/hf/Qwen2-1.5B-Instruct/",
        "MODEL_TYPE": "qwen_2",
        "ACT_TYPE": "bf16",
        "CONCURRENCY_LIMIT": "128",
    },
    main = "frontend_app_test.py",
    tags = ["frontend"],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
)
