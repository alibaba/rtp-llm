
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
    name = "kv_cache_config_pickle_test",
    srcs = [
        "kv_cache_config_pickle_test.py",
    ],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)
