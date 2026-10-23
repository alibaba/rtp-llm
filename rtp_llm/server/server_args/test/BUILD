py_test(
    name = "server_args_test",
    main = "server_args_test.py",
    srcs = ["server_args_test.py"],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
        "//rtp_llm/config/test:kv_cache_event_test_values",
        "//rtp_llm/server",
    ],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "server_args_test_rocm",
    main = "server_args_test.py",
    srcs = ["server_args_test.py"],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
        "//rtp_llm/config/test:kv_cache_event_test_values",
        "//rtp_llm/server",
    ],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
)
