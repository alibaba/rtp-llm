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
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        "//rtp_llm/server:server_args_util",
    ],
)
