load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

py_test(
    name = "vip_server_test",
    srcs = [
        "vip_server_test.py",
    ],
    deps = [
        test_requirement("pytest"),
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    env = {"GPU_COUNT": "0"},
    tags = [
        "L20_CU13",
        "frontend",
    ],
)

py_test(
    name = "frontend_start_test",
    srcs = [
        "frontend_app_test.py",
    ],
    main = "frontend_app_test.py",
    deps = [
        test_requirement("pytest"),
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
    env = {
        "CHECKPOINT_PATH": "/mnt/nas1/hf/Qwen2-1.5B-Instruct/",
        "MODEL_TYPE": "qwen_2",
        "ACT_TYPE": "bf16",
        "CONCURRENCY_LIMIT": "128",
    },
    tags = ["frontend"],
)