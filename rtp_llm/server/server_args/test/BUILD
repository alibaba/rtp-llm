py_test(
    name = "server_args_test",
    srcs = ["server_args_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "moe_config_pickle_test",
    srcs = ["moe_config_pickle_test.py"],
    # Config-only shared-library coverage; no GPU execution property is needed.
    data = [
        "//:th_transformer_config",
        "//rtp_llm:ops/libth_transformer_config.pyi",
    ],
    deps = [
        "//rtp_llm:config_ops",
    ],
)
