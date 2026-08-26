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
    name = "master_group_args_test",
    srcs = ["master_group_args_test.py"],
    deps = ["//rtp_llm/server:server"],
)
