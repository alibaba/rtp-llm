
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
    name = "grammar_tokenizer_info_test",
    srcs = [
        "grammar_tokenizer_info_test.py",
    ],
    deps = [
        "//rtp_llm:config",
    ],
)
