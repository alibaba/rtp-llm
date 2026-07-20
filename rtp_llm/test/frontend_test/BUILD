py_test(
    name = "vip_server_test",
    srcs = [
        "vip_server_test.py",
    ],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu': 'H20'},
    tags = ["frontend"],
)

py_test(
    name = "sleep_routes_test",
    srcs = [
        "sleep_routes_test.py",
    ],
    deps = [
        "//rtp_llm:fastapi",
        "//rtp_llm:frontend_sleep_routes",
        "//rtp_llm:frontend_worker_address_utils",
        "//rtp_llm:protobuf",
        "//rtp_llm:utils",
        "//rtp_llm/cpp/model_rpc:grpcio",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        "//rtp_llm/metrics:metrics",
    ],
    exec_properties = {'gpu': 'H20'},
    tags = ["frontend"],
)

py_test(
    name = "frontend_start_test",
    srcs = [
        "frontend_app_test.py",
    ],
    main = "frontend_app_test.py",
    deps = [
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
