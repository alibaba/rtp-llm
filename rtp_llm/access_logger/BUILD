py_library(
    name = "access_logger",
    srcs = glob([
        "*.py"
    ], exclude = ["*_test.py"]),
    visibility = ["//visibility:public"]
)

py_test(
    name = "async_log_handler_test",
    srcs = ["async_log_handler_test.py"],
    deps = [":access_logger", "//rtp_llm:testlib"],
)
