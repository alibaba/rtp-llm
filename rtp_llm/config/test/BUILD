
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

py_library(
    name = "kv_cache_event_test_values",
    srcs = [
        "kv_cache_event_test_values.py",
    ],
    visibility = ["//rtp_llm/server/server_args/test:__pkg__"],
)

py_test(
    name = "grammar_tokenizer_info_test",
    srcs = [
        "grammar_tokenizer_info_test.py",
    ],
    data = [
        "//:th_grammar_tokenizer_info",
    ],
    deps = [
        "//rtp_llm:config",
    ],
)

py_test(
    name = "kv_cache_config_pickle_test",
    srcs = [
        "kv_cache_config_pickle_test.py",
    ],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        ":kv_cache_event_test_values",
        "//rtp_llm:ops",
    ],
)

py_test(
    name = "output_vocab_config_test",
    srcs = ["output_vocab_config_test.py"],
    deps = [
        "//rtp_llm:config",
    ],
)
