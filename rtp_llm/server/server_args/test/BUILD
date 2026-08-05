py_test(
    name = "server_args_test",
    srcs = ["server_args_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "moe_config_pickle_test",
    srcs = ["moe_config_pickle_test.py"],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:config_ops",
    ],
)
