py_test(
    name = "server_args_test",
    srcs = ["server_args_test.py"],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm/config/test:kv_cache_event_test_values",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)
