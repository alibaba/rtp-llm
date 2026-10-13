py_test(
    name = "async_log_handler_test",
    srcs = ["async_log_handler_test.py"],
    deps = [
        "//rtp_llm/access_logger:access_logger",
    ],
)

py_test(
    name = "access_logger_test",
    srcs = ["access_logger_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:structure",
        "//rtp_llm:testlib",
        "//rtp_llm/access_logger:access_logger",
    ],
)
