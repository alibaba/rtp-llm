py_library(
    name = "access_logger",
    srcs = glob([
        "*.py"
    ], exclude = [
        "*_test.py",
    ]),
    deps = [
        "//rtp_llm:numpy",
        "//rtp_llm:structure",
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "access_logger_test",
    srcs = ["access_logger_test.py"],
    deps = [
        ":access_logger",
        "//rtp_llm:structure",
    ],
)
