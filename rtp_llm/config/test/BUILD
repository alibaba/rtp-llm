
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
    name = "model_config_override_test",
    srcs = ["model_config_override_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
        "//rtp_llm:transformers",
    ],
)

