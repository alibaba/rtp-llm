
py_test(
    name = "grammar_compile_threads_test",
    srcs = [
        "grammar_compile_threads_test.py",
    ],
    data = [
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "server_config_setup_test",
    srcs = [
        "server_config_setup_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)


py_test(
    name = "engine_config_test",
    srcs = ["engine_config_test.py"],
    data = ["//rtp_llm:sdk"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "SM103_CU13"},
)
