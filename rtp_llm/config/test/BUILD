
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
    name = "cache_event_subscriber_launcher_test",
    srcs = ["cache_event_subscriber_launcher_test.py"],
    data = ["//rtp_llm:cache_event_subscriber_launcher.py"],
)
